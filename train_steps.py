from models import Speaker,Listener,AlexNet,AlexNetDecoder,EvaluationHead,RegionHead
import jax
import jax.numpy as jnp
from functools import partial
import distrax
from jax import lax
import flax.linen as nn
from jax.lib import xla_bridge
import optax
from utils import gauss_kernel_2d

@partial(jax.jit, static_argnums=(7,8,9))
@partial(jax.value_and_grad, argnums=(0,1,4),has_aux=True)
def forward_lewis(encoder_params : jax.Array, speaker_params : jax.Array ,target_speaker_params : jax.Array,target_encoder_params : jax.Array, listener_params : jax.Array,images,key,num_distractors : int ,mode : int, training : bool):
    
    subkey,key = jax.random.split(key)
    target_images = AlexNet().apply(target_encoder_params,images,subkey,True)
    
    subkey,key = jax.random.split(key)
    images = AlexNet().apply(encoder_params,images,subkey,True)

    subkey,key = jax.random.split(key)
    (logits,values,action_logprobs,entropies,message) = Speaker(256,20,10,10).apply(speaker_params,images,mode,jnp.zeros((1,),dtype=jnp.int8),subkey)
    subkey,key = jax.random.split(key)
    (target_logits,_,_,_,_) = Speaker(256,20,10,10).apply(target_speaker_params,target_images,2,message,subkey)

    images_encoded,hidden_encoded = Listener(512,20,10,256).apply(listener_params,images,message)

    normalize = lambda x : x/jnp.linalg.norm(x,ord=2,axis=-1)[:,None]
    images_encoded = normalize(images_encoded)
    hidden_encoded = normalize(hidden_encoded)

    # (batch_size x batch_size)
    sim_matrix = -jnp.sum((images_encoded[:,None,:] - hidden_encoded)**2, axis=-1)

    distractor_indices = jnp.repeat(jnp.arange(start=0,stop=images.shape[0],dtype=jnp.int32)[None,:],repeats=images.shape[0],axis=0) 
    distractor_indices += jnp.triu(jnp.ones((images.shape[0],images.shape[0]),dtype=jnp.int32))
    distractor_indices = distractor_indices[:,:-1]

    subkey,key = jax.random.split(key)
    distractor_indices = jax.random.permutation(subkey,distractor_indices,axis=-1,independent=True)
    
    distractor_indices = distractor_indices[:,:num_distractors]
    target_distractor_indices = jnp.concatenate([jnp.arange(start=0,stop=images.shape[0])[:,None],distractor_indices],axis=-1)

    gather = jax.vmap(lambda x,index : x[index], in_axes=(0,0))
    target_distractor_similarities = gather(sim_matrix,target_distractor_indices)

    argmax = jnp.argmax(target_distractor_similarities,axis=-1)

    reward = jnp.where(argmax == 0,1,0)
    
    Lv = jnp.mean(jnp.square((reward[:,None] - values)),axis=-1)
    LPolicy = -jnp.mean((reward[:,None] - lax.stop_gradient(values))*action_logprobs,axis=-1)
    Lentropy = -0.0001*jnp.mean(entropies,axis=-1)
    LKlDiv = 0.5*jnp.mean(distrax.Softmax(logits).kl_divergence(distrax.Softmax(target_logits)),axis=-1)
    LReceiver = -jax.nn.log_softmax(target_distractor_similarities,axis=-1)[:,0]

    return (jnp.mean(Lv + LPolicy + Lentropy + LKlDiv + LReceiver)),(jnp.mean(reward),Lv,LPolicy,Lentropy,LKlDiv,LReceiver,values,entropies,target_distractor_similarities)


@partial(jax.jit, static_argnums=(4,))
@partial(jax.value_and_grad, argnums=(0,1),has_aux = True)
def forward_predict_transforms(encoder_params : jax.Array, classifier_layer_params: jax.Array,images,key,training : bool):

    # generate a random rotation target (0,1,2,3) -> (0,90,180,270)
    subkey,key = jax.random.split(key)
    targets = jax.random.randint(subkey,minval=0,maxval=4,shape=(images.shape[0],))
    targets_one_hot = jax.nn.one_hot(targets,4)

    def rot(img,k):
        return jnp.rot90(img,k=0)*k[0]+jnp.rot90(img,k=1)*k[1]+jnp.rot90(img,k=2)*k[2]+jnp.rot90(img,k=3)*k[3]
    
    # rotate each image by its rotation target
    images = jax.vmap(rot,(0,0))(images,targets_one_hot)
    
    subkey,key = jax.random.split(key)
    encoded = AlexNet().apply(encoder_params,images,subkey,training)

    transform_prediction = nn.Dense(4).apply(classifier_layer_params,encoded)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits=transform_prediction,labels=targets)

    accuracy = jnp.where(jnp.argmax(transform_prediction,axis=-1)==targets,1,0)*1.0

    return jnp.mean(loss),jnp.mean(accuracy)

@partial(jax.jit, static_argnums=(4,))
@partial(jax.value_and_grad, argnums=(0,1),has_aux = False)
def forward_autoencoder(encoder_params : jax.Array, decoder_params: jax.Array,images,key,training : bool):

    subkey,key = jax.random.split(key)
    encoded = AlexNet().apply(encoder_params,images,subkey,training)

    decoded = AlexNetDecoder().apply(decoder_params,encoded)

    loss = optax.l2_loss(decoded,images)

    return jnp.mean(loss)


@partial(jax.jit, static_argnums=(5,))
@partial(jax.value_and_grad, argnums=(0,1),has_aux = True)
def forward_evaluate(encoder_params : jax.Array, evaluation_dense_params: jax.Array,images,targets,key,training : bool):
    subkey,key = jax.random.split(key)
    encoded = AlexNet().apply(encoder_params,images,subkey,True)

    prediction = EvaluationHead(40).apply(evaluation_dense_params,encoded)

    loss = optax.sigmoid_binary_cross_entropy(prediction,targets)
    prediction = nn.sigmoid(prediction)
    prediction = jnp.where(prediction>=0.5,1,0)
    accuracy = jnp.where(prediction==targets,1,0)*1.0


    accuracy_1 = jnp.sum(accuracy*targets)/jnp.sum(targets)
    accuracy_0 = jnp.sum(accuracy*(1-targets))/jnp.sum(1-targets)

    return jnp.mean(loss),(jnp.mean(accuracy),jnp.mean(accuracy_0),jnp.mean(accuracy_1))
