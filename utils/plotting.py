import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, List, Union
from utils.base_utils import *

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from typing import Tuple, List, Union
from PIL import Image
import torchvision

def label_3d(ax):
    ax[0].set(xlabel='x', ylabel='y')
    ax[1].set(xlabel='x', ylabel='z')
    ax[2].set(xlabel='y', ylabel='z')

from cv2 import line
def show_image(x, idx):
    fig = plt.figure()
    plt.imshow(x[idx].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())

def savefig(name, savedir='figures'):
    plt.savefig(os.path.join(savedir, f'{name}.pdf'), bbox_inches='tight', dpi=300)

def savefig_eps(name, savedir='figures'):
    plt.savefig(os.path.join(savedir, f'{name}.eps'), bbox_inches='tight', dpi=300)
    
def plot_with_sem(x, y_mean, y_sem, figsize=(6, 4), **kwargs):
    '''Plot the mean of the data with the standard error of the mean'''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # print(*kwargs)
    ax.plot(x, y_mean, **kwargs)
    ax.fill_between(x, y_mean-y_sem, y_mean+y_sem, alpha=0.5)
    return fig, ax

def plot_errorbars(y_mean, y_sem, figsize=(6, 4), fmt='.', **kwargs):
    '''Plot the errorbars around each data point with no connecting line'''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if 'x' not in kwargs.keys():
        x = range(y_mean.shape[0])
    else:
        x = kwargs['x']
    ax.errorbar(x, y_mean, yerr=y_sem, fmt=fmt, **kwargs)
    return fig, ax

def plot_N_strips_of_M_images(N, M, images):
    for n in range(N):
        fig, ax = plt.subplots(1, M, figsize=(M*2, 2))
        for i, a in enumerate(ax):
            a.imshow(images[i+n*M].squeeze(), cmap='gray', vmin=0, vmax=1)
            a.axis('off')



def plot_color_img(img:Union[torch.Tensor, np.ndarray], 
                    figsize:Tuple[int, int] = (4, 4), 
                    unnormalize_to_01:bool = True,
                    title:Union[None, str, List[str]] = None, 
                    suptitle:Union[str, None] = None,
                    axis_off:bool = True):
    if isinstance(img, torch.Tensor):
        img = t2n(img)
    
    if unnormalize_to_01:
        # img = img * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
        img = (img + 1) / 2.0

    while img.ndim > 3:
        img = img.squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img.transpose(1, 2, 0), vmin=0, vmax=1)
    ax.set(xticks=[], yticks=[])
    if axis_off:
        ax.axis('off')
    if title is not None and isinstance(title, list):
        ax.set(title=title)
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig
    
def plot_N_strips_of_M_images(N, M, images, figsize=None):
    """Plot N strips of M images each"""
    if figsize is None:
        figsize = (M*2, 2)

    for n in range(N):
        fig, ax = plt.subplots(1, M, figsize=figsize)
        for i, a in enumerate(ax):
            a.imshow(images[i+n*M].squeeze(), cmap='gray', vmin=0, vmax=1)
            a.axis('off')
        fig.tight_layout()

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def make_presentable(imgs):
    '''unnormalize the images from [-1, 1] to [0, 1]'''
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    imgs = (imgs + 1) / 2.0

    # clip
    imgs = np.clip(imgs, 0, 1)

    # transpose with numpy
    if imgs.ndim == 4:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    elif imgs.ndim == 3:
        imgs = np.transpose(imgs, (1, 2, 0))
    # if imgs.ndim == 4:
    #     imgs = imgs.transpose(0, 2, 3, 1)
    # elif imgs.ndim == 3:
    #     imgs = imgs.transpose(1, 2, 0)

    return imgs

def plot_img(img:Union[torch.Tensor, np.ndarray], 
             figsize:Tuple[int, int] = (2, 2), 
             vmin:Union[None, float] = None, 
             vmax:Union[None, float] = None, 
             title:Union[None, str, List[str]] = None, 
             suptitle:Union[str, None] = None,
             axis_off:bool = True):
    if vmin==None:
        vmin = img.min().item()
        vmax = img.max().item()
    
    if isinstance(img, torch.Tensor):
        img = t2n(img)
    if img.ndim > 2:
        img = img.squeeze()
        if img.ndim > 2:
            num_imgs = img.shape[0]
            fig, ax = plt.subplots(1, num_imgs, figsize=figsize)
            for i in range(num_imgs):
                ax[i].imshow(img[i], cmap='gray', vmin=vmin, vmax=vmax)
                ax[i].set(xticks=[], yticks=[])
                if axis_off:
                    ax[i].axis('off')
                if title is not None and isinstance(title, list) and len(title) == num_imgs:
                    ax[i].set(title=title[i])
            if suptitle is not None:
                fig.suptitle(suptitle)
            return fig
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _ = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax);
    ax.set(xticks=[], yticks=[])
    if axis_off:
        ax.axis('off')
    if title is not None:
        ax.set(title=title)
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig

def plot_N_strips_of_M_images(N, M, images):
    for n in range(N):
        fig, ax = plt.subplots(1, M, figsize=(M*2, 2))
        for i, a in enumerate(ax):
            a.imshow(images[i+n*M].squeeze(), cmap='gray', vmin=0, vmax=1)
            a.axis('off')

def plot_img(img:Union[torch.Tensor, np.ndarray], 
             figsize:Tuple[int, int] = (2, 2), 
             vmin:Union[None, float] = None, 
             vmax:Union[None, float] = None, 
             title:Union[None, str, List[str]] = None, 
             suptitle:Union[str, None] = None,
             axis_off:bool = True):
    if vmin==None:
        vmin = img.min().item()
        vmax = img.max().item()
    
    if isinstance(img, torch.Tensor):
        img = t2n(img)
    if img.ndim > 2:
        img = img.squeeze()
        if img.ndim > 2:
            num_imgs = img.shape[0]
            fig, ax = plt.subplots(1, num_imgs, figsize=figsize)
            for i in range(num_imgs):
                ax[i].imshow(img[i], cmap='gray', vmin=vmin, vmax=vmax)
                ax[i].set(xticks=[], yticks=[])
                if axis_off:
                    ax[i].axis('off')
                if title is not None and isinstance(title, list) and len(title) == num_imgs:
                    ax[i].set(title=title[i])
            if suptitle is not None:
                fig.suptitle(suptitle)
            return fig
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _ = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax);
    ax.set(xticks=[], yticks=[])
    if axis_off:
        ax.axis('off')
    if title is not None:
        ax.set(title=title)
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig
