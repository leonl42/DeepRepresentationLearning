import jax.numpy as jnp
import flax.linen as nn
import jax
from jax import lax
import flax
import distrax
from functools import partial
import optax


class AlexNet(nn.Module):

    @nn.compact
    def __call__(self, x,key, training):

        x = nn.Conv(features=64,kernel_size=(11,11),strides=(4,4),padding="VALID")(x)

        x = nn.relu(x)
        x = nn.max_pool(x,window_shape=(3,3),strides=(2,2))

        x = nn.Conv(features=128,kernel_size=(5,5),padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x,window_shape=(3,3),strides=(2,2))

        x = nn.Conv(features=256,kernel_size=(3,3),padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(features=256,kernel_size=(3,3),padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(features=128,kernel_size=(3,3),padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x,window_shape=(3,3),strides=(2,2))

        x = x.reshape(x.shape[0],-1)

        x = nn.Dense(2048)(x)
        x = nn.relu(x)
        #subkey,key = jax.random.split(key)
        #x = nn.Dropout(0.5)(x, deterministic=not training,rng=subkey)

        x = nn.Dense(2048)(x)
        x = nn.relu(x)
        #subkey,key = jax.random.split(key)
        #x = nn.Dropout(0.5)(x, deterministic=not training,rng=subkey)

        return x
    

class AlexNetDecoder(nn.Module):

    @nn.compact
    def __call__(self, x):

        batch_size = x.shape[0]

        x = nn.Dense(3200)(x)
        x = nn.relu(x)

        x = x.reshape(-1,5,5,128)

        x = jax.image.resize(x,(batch_size,10,10,256),method="bilinear")
        x = nn.Conv(features=32,kernel_size=(3,3),strides=(1,1),padding=(1,1))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(batch_size,20,20,x.shape[-1]),method="bilinear")
        x = nn.Conv(features=64,kernel_size=(3,3),strides=(1,1),padding=(1,1))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(batch_size,40,40,x.shape[-1]),method="bilinear")
        x = nn.Conv(features=128,kernel_size=(3,3),strides=(1,1),padding=(1,1))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(batch_size,80,80,x.shape[-1]),method="bilinear")
        x = nn.Conv(features=128,kernel_size=(3,3),strides=(1,1),padding=(1,1))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(batch_size,160,160,x.shape[-1]),method="bilinear")
        x = nn.Conv(features=128,kernel_size=(3,3),strides=(1,1),padding=(1,1))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(batch_size,178,178,x.shape[-1]),method="bilinear")
        x = nn.Conv(features=3,kernel_size=(3,3),strides=(1,1),padding=(1,1))(x)


        """
        x = nn.ConvTranspose(features=128,kernel_size=(11,11),padding="VALID",strides=(4,4))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=128,kernel_size=(5,5),padding="VALID",strides=(3,3))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(x.shape[0],86,86,256),method="bilinear")

        x = nn.ConvTranspose(features=256,kernel_size=(3,3),padding="VALID",strides=(2,2))(x)
        x = nn.relu(x)

        x = jax.image.resize(x,(x.shape[0],174,174,256),method="bilinear")

        x = nn.ConvTranspose(features=3,kernel_size=(5,5),padding="VALID")(x)
        x = nn.relu(x)
        """

        return x


class EvaluationHead(nn.Module):
    output_size : int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_size)(x)
    

class RegionHead(nn.Module):

    @nn.compact
    def __call__(self, x, key, training):

        x = AlexNet()(x, key, training)
        
        return nn.Dense(2)(x)

class HaikuLSTMCell(nn.Module):

    hidden_size : int

    @nn.compact
    def __call__(self, x, state):
        h, c = state
        x_and_h = jnp.concatenate([x, h], axis=-1)
        gated = nn.Dense(features=4*self.hidden_size, name="cell_dense")(x_and_h)

        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.

        c = f * c + jax.nn.sigmoid(i) * jax.nn.tanh(g)
        h = jax.nn.sigmoid(o) * jax.nn.tanh(c)
        return h, c

class Speaker(nn.Module):

    hidden_size : int # 256
    vocabulary_size : int # 20
    symbol_embedding_size : int # 10
    max_message_length : int # 10

    @nn.compact
    def __call__(self, images : jax.Array, mode : str, message : jax.Array, key):

        if message==None:
            message == jnp.zeros((images.shape[0],10))
        
        encoded = nn.Dense(2*self.hidden_size, name="encoder")(images)
        (h,c) = jnp.split(encoded,indices_or_sections=2,axis = -1)

        symbol = jnp.ones((encoded.shape[0]),dtype=jnp.int32)*self.vocabulary_size

        message_logits = []
        message_action_logprobs = []
        message_entropies = []
        message_symbols = []
        message_values = []

        cell = HaikuLSTMCell(self.hidden_size, name="cell")
        symbol_embedder = nn.Embed(self.vocabulary_size+1,self.symbol_embedding_size,name="symbol_embedder")
        policy_head = nn.Dense(self.vocabulary_size, name="policy_head")
        value_head = nn.Dense(1, name="value_head")

        for i in range(self.max_message_length):

            cell_input = symbol_embedder(symbol)

            (h,c) = cell(cell_input,(h,c))

            logits = policy_head(h)
            value = value_head(h).squeeze(-1)

            distr = distrax.Softmax(logits=logits)

            if mode == 0:
                subkey,key = jax.random.split(key)
                symbol = distr.sample(seed=subkey)
            elif mode == 1:
                symbol = jnp.argmax(logits, axis = -1)
            elif mode == 2:
                symbol = message[:,i]

            action_logprop = distr.log_prob(symbol)
            entropy = distr.entropy()

            message_logits.append(logits)
            message_values.append(value)
            message_action_logprobs.append(action_logprop)
            message_entropies.append(entropy)
            message_symbols.append(symbol)
           

        return (jnp.stack(message_logits,axis=1),
                jnp.stack(message_values,axis=1),
                jnp.stack(message_action_logprobs,axis=1),
                jnp.stack(message_entropies,axis=1),
                jnp.stack(message_symbols,axis=1))

            

            
class Listener(nn.Module):

    hidden_size : int # 512
    vocabulary_size : int # 20
    symbol_embedding_size : int # 10
    output_size : int # 256

    @nn.compact
    def __call__(self, images : jax.Array, message : jax.Array):
        
        message_embedded = nn.Embed(self.vocabulary_size,self.symbol_embedding_size, name="symbol_embedder")(message)

        cell = HaikuLSTMCell(self.hidden_size, name = "cell")
        h = jnp.zeros((message.shape[0],self.hidden_size))
        c = jnp.zeros((message.shape[0],self.hidden_size))
        for i in range(message_embedded.shape[1]):
            cell_input = message_embedded[:,i,:]
            (h,c) = cell(cell_input,(h,c))

        images_encoded = nn.Dense(self.output_size, name = "image_encoder")(images)
        hidden_encoded = nn.Dense(self.output_size, name = "hidden_encoder")(h)

        return images_encoded,hidden_encoded


