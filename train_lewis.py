import os
import sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

from dataloader import load_dataset_celeb_a,load_image
from models import Speaker,Listener,AlexNet,EvaluationHead
import jax
from tqdm import tqdm
import copy
from utils import AuxAgg
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
import pickle as pkl
from train_steps import forward_lewis,forward_evaluate
import distrax
from jax import lax
from jax.lib import xla_bridge
import optax
import sys
print("using backend: ", xla_bridge.get_backend().platform)

# initialize key to make training somewhat deterministic
SEED = int(sys.argv[1])
NUM_DISTRACTORS = int(sys.argv[2])
key = jax.random.key(SEED)

# create dataset splits
print("loading dataset...")

image_paths,labels = load_dataset_celeb_a("./dataset/celeb_a/list_attr_celeba.csv","./dataset/celeb_a/img_align_celeba/img_align_celeba/")
ds = tf.data.Dataset.from_tensor_slices((image_paths,labels))
ds = ds.map(load_image)
dataset_train = ds.take(int(len(ds)*0.8)).shuffle(buffer_size=64000).batch(3072).prefetch(tf.data.AUTOTUNE)
dataset_valid = ds.skip(int(len(ds)*0.8)).take(int(len(ds)*0.1)).shuffle(buffer_size=8000).batch(3072).prefetch(tf.data.AUTOTUNE)
dataset_test = ds.skip(int(len(ds)*0.8)).skip(int(len(ds)*0.1)).shuffle(buffer_size=8000).batch(3072).prefetch(tf.data.AUTOTUNE)

print("initializing models...")
# Initialize model and optimizer params

subkey,key = jax.random.split(key)
evaluation_dense_params = EvaluationHead(40).init(subkey,jnp.ones((2,2048)))

subkey,key = jax.random.split(key)
alexnet_params = AlexNet().init(subkey,jnp.ones((10,178,178,3)),jax.random.key(0),True)

subkey,key = jax.random.split(key)
speaker_params = Speaker(256,20,10,10).init(subkey,jnp.ones((2,2048)),0,jnp.zeros((2,1),dtype=jnp.int8),jax.random.key(0))

subkey,key = jax.random.split(key)
target_speaker_params = Speaker(256,20,10,10).init(subkey,jnp.ones((2,2048)),0,jnp.zeros((2,1),dtype=jnp.int8),jax.random.key(0))

subkey,key = jax.random.split(key)
listener_params = Listener(512,20,10,256).init(subkey,jnp.ones((2,2048)),jnp.zeros((2,1),dtype=jnp.int8))

evaluation_dense_optimizer = optax.adam(learning_rate=0.001)
evaluation_dense_opt_params = evaluation_dense_optimizer.init(evaluation_dense_params)

alexnet_optimizer= optax.adam(learning_rate=0.0001)
alexnet_opt_params = alexnet_optimizer.init(alexnet_params)

speaker_optimizer= optax.adam(learning_rate=0.0001)
speaker_opt_params = speaker_optimizer.init(speaker_params)

listener_optimizer= optax.adam(learning_rate=0.0001)
listener_opt_params = listener_optimizer.init(listener_params)

auxagg = AuxAgg()
best_alexnet_params = copy.deepcopy(alexnet_params)
best_valid_acc = 0
last_best_valid_acc_update = 0

print("starting training...")
os.makedirs("./saves/train_lewis_seed_" + str(SEED) + "_" + str(NUM_DISTRACTORS) + "/")

for epoch in tqdm(range(10000)):
    for image,_ in dataset_valid.as_numpy_iterator():

        #######################################
        ########## validation step ############
        #######################################

        subkey,key = jax.random.split(key)
        (loss,(reward,Lv,LPolicy,Lentropy,LKlDiv,LReceiver,values,entropies,target_distractor_similarities)),_ = forward_lewis(alexnet_params,speaker_params,target_speaker_params,listener_params,image,subkey,NUM_DISTRACTORS,1,False)
        auxagg.add(epoch,["valid_loss","valid_acc"],[loss,reward])

    for image,_ in dataset_train.as_numpy_iterator():

        subkey,key = jax.random.split(key)
        (loss,(reward,Lv,LPolicy,Lentropy,LKlDiv,LReceiver,values,entropies,target_distractor_similarities)),grads= forward_lewis(alexnet_params,speaker_params,target_speaker_params,listener_params,image,subkey,NUM_DISTRACTORS,0,True)
        auxagg.add(epoch,["train_loss","train_acc"],[loss,reward])

        updates, alexnet_opt_params = alexnet_optimizer.update(grads[0], alexnet_opt_params)
        alexnet_params = optax.apply_updates(alexnet_params, updates)

        updates, speaker_opt_params = speaker_optimizer.update(grads[1], speaker_opt_params)
        speaker_params = optax.apply_updates(speaker_params, updates)

        updates, listener_opt_params = listener_optimizer.update(grads[2], listener_opt_params)
        listener_params = optax.apply_updates(listener_params, updates)

        target_speaker_params = jax.tree_map(lambda x,y : 0.99*x+0.01*y,target_speaker_params,speaker_params)

    auxagg.collapse(epoch)
    auxagg.save("./saves/train_lewis_seed_" + str(SEED) + "_" + str(NUM_DISTRACTORS) + "/","train_auxagg.pkl")

    if auxagg.mean(epoch,["valid_acc"])[0] > best_valid_acc:
        best_valid_acc = auxagg.mean(epoch,["valid_acc"])[0]
        last_best_valid_acc_update = epoch
        best_alexnet_params = copy.deepcopy(alexnet_params)
    if epoch - last_best_valid_acc_update >= 10 and auxagg.mean(epoch,["train_acc"])[0] >= 0.8:
        break
    
print("-------- evaluating ---------")

best_valid_loss = jnp.inf
last_best_valid_loss_update = 0

dataset_train = ds.take(int(len(ds)*0.8)).shuffle(buffer_size=64000).batch(32).prefetch(tf.data.AUTOTUNE)
dataset_valid = ds.skip(int(len(ds)*0.8)).take(int(len(ds)*0.1)).shuffle(buffer_size=8000).batch(32).prefetch(tf.data.AUTOTUNE)
dataset_test = ds.skip(int(len(ds)*0.8)).skip(int(len(ds)*0.1)).shuffle(buffer_size=8000).batch(32).prefetch(tf.data.AUTOTUNE)


auxagg = AuxAgg()

for epoch in tqdm(range(1000)):

    #######################################
    ########## validation step ############
    #######################################

    for image,labels in dataset_valid.as_numpy_iterator():
        subkey,key = jax.random.split(key)
        (loss,(accuracy,accuracy_0,accuracy_1)),_ = forward_evaluate(best_alexnet_params,evaluation_dense_params,image,labels,subkey,False)

        auxagg.add(epoch, ["valid_loss","valid_acc","valid_acc_0","valid_acc_1"],[loss,accuracy,accuracy_0,accuracy_1])

    #######################################
    ########## train step ############
    #######################################

    for image,labels in dataset_train.as_numpy_iterator():
        subkey,key = jax.random.split(key)
        (loss,(accuracy,accuracy_0,accuracy_1)),grads = forward_evaluate(best_alexnet_params,evaluation_dense_params,image,labels,subkey,True)

        auxagg.add(epoch, ["train_loss","train_acc","train_acc_0","train_acc_1"],[loss,accuracy,accuracy_0,accuracy_1])

        updates, evaluation_dense_opt_params = evaluation_dense_optimizer.update(grads[1], evaluation_dense_opt_params)
        evaluation_dense_params = optax.apply_updates(evaluation_dense_params, updates)

    auxagg.collapse(epoch)
    auxagg.save("./saves/train_lewis_seed_" + str(SEED) + "_" + str(NUM_DISTRACTORS) + "/","evaluate_auxagg.pkl")

    if auxagg.mean(epoch,["valid_loss"])[0] < best_valid_loss:
        best_valid_loss = auxagg.mean(epoch,["valid_loss"])[0]
        last_best_valid_loss_update = epoch

    if epoch - last_best_valid_loss_update >= 10:
        break
