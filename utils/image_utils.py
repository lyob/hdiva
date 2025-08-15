from PIL import Image
import numpy as np
import torch

def resize_image(im, s):
    image_pil = Image.fromarray(im) # data type needed is uint8
    newsize = (int(image_pil.size[0] * s), int(image_pil.size[1] * s))
    image_pil_resize = image_pil.resize(newsize, resample=Image.BICUBIC)
    image_re = np.array(image_pil_resize)
    return image_re

def rgb_to_gray(data):
    n, h, w, c = data.shape
    return data.mean(3).reshape(n,h,w,1)

def change_intensity(  im , k):
    temp = np.zeros((k,im.shape[0], im.shape[1], im.shape[2])).astype('float32')
    for i in range(k):
        temp[i] = np.random.rand(1).astype('float32') * im
    return temp

def change_intensity_dataset(dataset, k):
    temp = []
    for im in dataset:
        temp.append(change_intensity(im,k))
    return np.concatenate(temp)

def int_to_float(X):    
    if X.dtype == 'uint8':
        return (X/255).astype('float32')
    else:
        return X

# @dataclass
class NoiseLevel():
    def __init__(self, t):
        assert t < 1000, 't should be less than 1000'
        assert t > -2, 't should be -1 or non-negative'
        self.t: int = t

def noise(autoprior, clean_image:torch.Tensor, noise_level:NoiseLevel, device=torch.device('cuda')):
    '''add noise to a clean image'''
    t = noise_level.t
    if t == -1:
        return clean_image
    timestep = torch.tensor([t]).repeat_interleave(1, dim=0).long().to(device)
    noisy_image = autoprior.make_noisy(clean_image, timestep)[0]
    return noisy_image
