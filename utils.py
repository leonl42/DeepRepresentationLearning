import jax
import jax.numpy as jnp
import pickle
import os

class AuxAgg:
    """
    Stores a bunch of data. Can also plot stuff
    """

    def __init__(self) -> None:
        self.data_dict = {}
    
    def add(self, epoch, keys, data_list):

        if not epoch in self.data_dict:
            self.data_dict[epoch] = {}

        for key,data_element in zip(keys,data_list):

            if key in self.data_dict[epoch]:
                self.data_dict[epoch][key].append(data_element)

            else:
                self.data_dict[epoch][key] = [data_element]

    def collapse(self, epoch):
        for key in self.data_dict[epoch]:
            self.data_dict[epoch][key] = [jnp.mean(jnp.stack(self.data_dict[epoch][key]))]

    def mean(self, epoch, keys):
        means = []
        for key in keys:
            mean = jnp.mean(jnp.stack(self.data_dict[epoch][key]))
            means.append(mean)

        return means
    
    def max_mean(self, keys):
        saved_means = [[] for _ in keys]
        for epoch in self.data_dict:
            means = self.mean(epoch,keys)
            for saved_mean,mean in zip(saved_means,means):
                saved_mean.append(mean)
        return [(saved_mean.index(max(saved_mean)),max(saved_mean)) for saved_mean in saved_means]
    
    def min_mean(self, keys):
        saved_means = [[] for _ in keys]
        for epoch in self.data_dict:
            means = self.mean(epoch,keys)
            for saved_mean,mean in zip(saved_means,means):
                saved_mean.append(mean)
        return [(saved_mean.index(min(saved_mean)),min(saved_mean)) for saved_mean in saved_means]


    def save(self, path, file):
        os.makedirs(path,exist_ok=True)
        with open(path + file, "wb") as f:
            pickle.dump(self.data_dict,f)

    def load(self, path, file):
        with open(path + file, "rb") as f:
            self.data_dict = pickle.load(f)


def gauss_kernel_2d(center_x,center_y,sigma,img_x,img_y):

    batch_size = center_x.shape[0]

    img_arr_x = jnp.arange(0,img_y,)
    img_arr_y = jnp.arange(0,img_x)
    a,b = jnp.meshgrid(img_arr_y,img_arr_x)
    a = jnp.repeat(a[None,:,:],repeats=batch_size,axis=0)
    b = jnp.repeat(b[None,:,:],repeats=batch_size,axis=0)

    exponent = -((center_x[:,None,None]-a)**2 + (center_y[:,None,None]-b)**2)/(2*sigma[:,None,None])
    kernel = jnp.exp(exponent)/(2*jnp.pi*(2*sigma[:,None,None])**2)
    kernel = kernel/jnp.max(kernel)
    return kernel

    