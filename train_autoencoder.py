import os
import sys
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

from dataloader import load_dataset_celeb_a,load_image
from models import Speaker,Listener,AlexNet,EvaluationHead,AlexNetDecoder
import jax
from tqdm import tqdm
from utils import AuxAgg
import pickle
import copy
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
import pickle as pkl
from train_steps import forward_autoencoder,forward_evaluate
import distrax
from jax import lax
from jax.lib import xla_bridge
import optax
import sys
print("using backend: ", xla_bridge.get_backend().platform)
import time



# initialize key to make training somewhat deterministic
SEED = int(sys.argv[1])
key = jax.random.key(SEED)

# create dataset splits
print("loading dataset...")

image_paths,labels = load_dataset_celeb_a("./dataset/celeb_a/list_attr_celeba.csv","./dataset/celeb_a/img_align_celeba/img_align_celeba/")

ds = tf.data.Dataset.from_tensor_slices((image_paths,labels))
ds = ds.map(load_image)


dataset_train = ds.take(int(len(ds)*0.8)).shuffle(buffer_size=8000).batch(32).prefetch(tf.data.AUTOTUNE)
dataset_valid = ds.skip(int(len(ds)*0.8)).take(int(len(ds)*0.1)).shuffle(buffer_size=2000).batch(32).prefetch(tf.data.AUTOTUNE)
dataset_test = ds.skip(int(len(ds)*0.8)).skip(int(len(ds)*0.1)).shuffle(buffer_size=2000).batch(32).prefetch(tf.data.AUTOTUNE)

print("initializing models...")
# Initialize model and optimizer params

subkey,key = jax.random.split(key)
evaluation_dense_params = EvaluationHead(40).init(subkey,jnp.ones((2,2048)))

subkey,key = jax.random.split(key)
alexnet_params = AlexNet().init(subkey,jnp.ones((10,178,178,3)),jax.random.key(0),True)

subkey,key = jax.random.split(key)
alexnetdecoder_params = AlexNetDecoder().init(subkey,jnp.ones((10,2048)))

evaluation_dense_optimizer = optax.adam(learning_rate=0.0001)
evaluation_dense_opt_params = evaluation_dense_optimizer.init(evaluation_dense_params)

alexnet_optimizer= optax.adam(learning_rate=0.0001)
alexnet_opt_params = alexnet_optimizer.init(alexnet_params)

alexnetdecoder_optimizer = optax.adam(learning_rate=0.0001)
alexnetdecoder_opt_params = alexnetdecoder_optimizer.init(alexnetdecoder_params)

auxagg = AuxAgg()

print("starting training...")

best_valid_loss = jnp.inf
best_alexnet_params = copy.deepcopy(alexnet_params)
last_best_valid_loss_update = 0

os.makedirs("./saves/train_autoencoder_seed_" + str(SEED) + "/")

for epoch in tqdm(range(1000)):

    
    #######################################
    ########## validation step ############
    #######################################

    for image,_ in tqdm(dataset_valid.as_numpy_iterator()):

        subkey,key = jax.random.split(key)
        loss,_ = forward_autoencoder(alexnet_params,alexnetdecoder_params,image,subkey,False)
        auxagg.add(epoch,["valid_loss"],[loss])

    #######################################
    ########## train step ############
    #######################################
        
    for image,_ in tqdm(dataset_train.as_numpy_iterator()):

        subkey,key = jax.random.split(key)
        loss,grads = forward_autoencoder(alexnet_params,alexnetdecoder_params,image,subkey,True)
        auxagg.add(epoch,["train_loss"],[loss])

        updates, alexnet_opt_params = alexnet_optimizer.update(grads[0], alexnet_opt_params)
        alexnet_params = optax.apply_updates(alexnet_params, updates)

        updates, alexnetdecoder_opt_params = alexnetdecoder_optimizer.update(grads[1], alexnetdecoder_opt_params)
        alexnetdecoder_params = optax.apply_updates(alexnetdecoder_params, updates)

    
    if auxagg.mean(epoch,["valid_loss"])[0] < best_valid_loss:
        best_valid_loss = auxagg.mean(epoch,["valid_loss"])[0]
        last_best_valid_loss_update = epoch
        best_alexnet_params = copy.deepcopy(alexnet_params)

    auxagg.collapse(epoch)
    auxagg.save("./saves/train_autoencoder_seed_" + str(SEED) + "/","train_auxagg.pkl")
    with open("./saves/train_autoencoder_seed_" + str(SEED) + "/" + "train_save.pkl", "wb") as f:
        pickle.dump((epoch,alexnet_params,alexnetdecoder_params,alexnet_opt_params,alexnetdecoder_opt_params,best_alexnet_params),f)

    if epoch - last_best_valid_loss_update >= 10:
        break


print("-------- evaluating ---------")

best_valid_loss = jnp.inf
last_best_valid_loss_update = 0


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

    if auxagg.mean(epoch,["valid_loss"])[0] < best_valid_loss:
        best_valid_loss = auxagg.mean(epoch,["valid_loss"])[0]
        last_best_valid_loss_update = epoch


    auxagg.collapse(epoch)
    auxagg.save("./saves/train_autoencoder_seed_" + str(SEED) + "/","evaluate_auxagg.pkl")

    if epoch - last_best_valid_loss_update >= 10:
        break
