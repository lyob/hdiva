import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from typing import Any, Tuple, List, Union
from numpy.typing import NDArray

from utils.plotting import plot_img
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import DataLoader
from functools import reduce
import operator
from utils.image_utils import noise, NoiseLevel

    
def select_target_img_subset(dataset, img_idx=-1, seed=4, plot=False, device=torch.device('cpu')):
    '''generate target image from test subset of the celeba dataset'''
    torch.manual_seed(seed)
    np.random.seed(seed)

    if img_idx < 0:
        img_idx = np.random.randint(0, len(dataset))

    DataLoader = torch.utils.data.DataLoader
    inference_batch_size = int(200)
    data_loader = DataLoader(dataset=dataset, batch_size=inference_batch_size, shuffle=True)
    # data_loader = DataLoader(dataset=dataset, batch_size=dataset.shape[0], shuffle=True)
    data = next(iter(data_loader))
    data = data.to(device)
    # data = data_loader.dataset
    target_image = data[img_idx].unsqueeze(0).to(device)
    
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(target_image.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set(xticks=[], yticks=[])
    
    return target_image


def check_idx_array(data:Union[List, NDArray[np.int_], Any]):
    if isinstance(data, list):
        if not all(isinstance(x, int) for x in data):
            raise TypeError("List must contain integers")
    elif isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.int_):
            raise TypeError("NumPy array must contain integers")
    else:
        raise TypeError("Unsupported data type, must be list or numpy array")

def select_target_img(dataset, img_idx:Union[List, NDArray[np.int_], int, np.int_]=-1, seed:int=4, plot:bool=False, scale:bool=False, device:torch.device=torch.device('cpu')):
    '''generate target image from test subset of the celeba dataset'''
    torch.manual_seed(seed)
    
    def select_img(dataset, img_idx):
        data_loader = DataLoader(dataset=dataset, batch_size=dataset.shape[0], shuffle=True)
        data = next(iter(data_loader))
        data = data.to(device)
        imgs = data[img_idx]
        if type(img_idx) == int or type(img_idx) == np.int_:
            return imgs.unsqueeze(0).to(device)
        else:
            return imgs.to(device)
    
    if (type(img_idx) == int or type(img_idx) == np.int_):
        if img_idx < 0:
            img_idx = np.random.randint(0, len(dataset))
    else:
        # replace any negative indices with random indices, with no replacement
        img_idx = np.array(img_idx)
        num_negatives = np.sum(img_idx < 0)
        img_idx[img_idx < 0] = np.random.choice(len(dataset), num_negatives, replace=False)
            
    selected_images = select_img(dataset, img_idx)

    if plot:
        plot_img(selected_images.cpu(), vmin=0, vmax=1)
    
    if scale:
        selected_images = selected_images * 2 - 1
    
    return selected_images

def select_target_imgs(dataset, seed=4, num_images=10, plot=False, device=torch.device('cpu')):
    '''generate target image from test subset of the celeba dataset'''
    torch.manual_seed(seed)
    np.random.seed(seed)

    # inference_batch_size = int(200)
    DataLoader = torch.utils.data.DataLoader
    data_loader = DataLoader(dataset=dataset, batch_size=dataset.shape[0], shuffle=True)
    # data = data_loader.dataset  # the order of this is not affected by the seed
    data = next(iter(data_loader))
    data = data.to(device)
    
    # choose 10 random images with no replacement
    img_idx = np.random.choice(len(data), num_images, replace=False)
    target_images = data[img_idx].to(device)
    
    if plot and num_images == 10:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        for ii in range(2):
            for jj in range(5):
                ax[ii, jj].imshow(target_images[5*ii +jj].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                ax[ii, jj].axis('off')
    
    return target_images

def select_and_noise_img(autoprior, dataset, img_idx, img_seed, t, device, rescale=True):
    img = select_target_img(dataset, img_idx=img_idx, seed=img_seed, plot=False, scale=rescale, device=device);
    img = noise(autoprior, img, NoiseLevel(t))
    return img 

import pandas as pd
def get_testset_attributes(trainset_size:int=50000):
    attribute_labels = (
        "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
        "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
        "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
        "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
        "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
        "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
        "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
    ).split()

    list_attr_celeba = 'datasets/celeba/list_attr_celeba.txt'
    # df_attributes = pd.read_csv(list_attr_celeba, sep='\s+', header=1, index_col=0)
    df_attributes = pd.read_csv(list_attr_celeba, sep='\s+', skiprows=1)

    # get all attributes after index 50000
    df_testset_attributes = df_attributes.iloc[trainset_size:]

    # print(df_attributes.head())
    # print(df_attributes.index)
    
    return (attribute_labels, df_testset_attributes)


def get_image_by_index(idx:int|np.ndarray|list, dataset_resolution:int=40):
    data_path = os.path.join('datasets/celeba', f'attribute_images_{dataset_resolution}x{dataset_resolution}.pt')
    dataset = torch.load(data_path, map_location='cpu')
    
    img = dataset[idx]
    return img

def get_image_by_name(name:str):
    idx = int(name.split('.')[0])
    img = get_image_by_index(idx-1)
    return img


def get_images_by_attribute(df, dataset, attributes:list|str, values:list|int) -> Tuple[np.ndarray, np.ndarray, torch.utils.data.Dataset]:
    '''Get all images with the specified attribute
    Args:
    - df: pandas DataFrame with attribute labels
    - dataset: torch.utils.data.Dataset with images
    - attributes: (list of) attribute name(s)
    - values: (list of) attribute value(s). 
    Returns:
    - image_indices: numpy array of image indices, e.g. 0, 1, 2, ...
    - image_names: numpy array of image "names", e.g. 000001.jpg
    - dataset: torch.utils.data.Dataset of images
    '''
    # get all image indices with the specified attribute
    
    all_attributes = (
        "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
        "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
        "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
        "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
        "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
        "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
        "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
    ).split()
    
    def check_types(attributes, values) -> Tuple[list, list]:
        if isinstance(attributes, str) and isinstance(values, int):
            assert attributes in all_attributes, "Attribute must exist"
            attributes = [attributes]
            values = [values]
        
        elif isinstance(attributes, str) and isinstance(values, list):
            assert isinstance(attributes, list), "Attributes must also be a list"
            
        elif isinstance(attributes, list):
            assert all(attr in all_attributes for attr in attributes), "All attribute must exist"
            if isinstance(values, int):
                # values = list(np.ones(len(attributes)) * values)
                values = list(np.repeat(values, len(attributes)))
        else:
            raise TypeError("Invalid type")
        return attributes, values
    

    def combine_conditions(*conditions):
        if not conditions:
            raise ValueError("No conditions provided")
        return reduce(operator.and_, conditions)
    
    attributes, values = check_types(attributes, values)

    conditions = []
    for aidx, attribute in enumerate(attributes):
        conditions.append(df[attribute] == values[aidx])
    conditions = combine_conditions(*conditions) if len(conditions) > 1 else conditions[0]
    
    # get all images with the specified attribute
    image_indices = np.where(conditions)[0]
    # image_indices = image_indices.astype(int)
    image_indices = [int(value) for value in image_indices]
    image_names = np.array(df[conditions].index)
    
    print(f"Number of images that satisfy condition: {len(image_indices)}")
    
    return np.array(image_indices), image_names, dataset[image_indices]


def get_images_by_attribute_dict(df, dataset, attribute_dict:dict):
    image_indices, _, dataset[image_indices] = get_images_by_attribute(df, dataset, attribute_dict['attributes'], attribute_dict['values'])
    return image_indices, dataset[image_indices]

def find_idx_of_image(dataset:torch.utils.data.Dataset, img:torch.Tensor):
    '''Find the index of an image in a dataset'''
    for i, data in enumerate(dataset):
        if torch.all(torch.eq(data, img)):
            return i
    return -1


# double checking that the ordering is correct
def double_check_ordering(test_subset:torch.utils.data.Dataset, trainset_size:int):
    names = [f'{i:06d}.jpg' for i in range(trainset_size+1, trainset_size+11)]
    fig, ax = plt.subplots(5, 2, figsize=(6, 12))
    for i, a in enumerate(ax):
        a[0].imshow(test_subset[i].squeeze(), cmap='gray', vmin=0, vmax=1);
        a[0].set(title=f'idx {i}')
        a[0].axis('off')
        
        a[1].imshow(get_image_by_name(names[i]).squeeze(), cmap='gray', vmin=0, vmax=1);
        a[1].set(title=names[i])
        a[1].axis('off')


from torch.utils.data import DataLoader
def select_target_img(dataset, img_idx:Union[List, NDArray[np.int_], int, np.int_]=-1, seed:int=4, plot:bool=False, scale:bool=False, device:torch.device=torch.device('cpu')):
    '''generate target image from test subset of the celeba dataset'''
    torch.manual_seed(seed)
    
    def select_img(dataset, img_idx):
        data_loader = DataLoader(dataset=dataset, batch_size=dataset.shape[0], shuffle=True)
        data = next(iter(data_loader))
        data = data.to(device)
        imgs = data[img_idx]
        if type(img_idx) == int or type(img_idx) == np.int_:
            return imgs.unsqueeze(0).to(device)
        else:
            return imgs.to(device)
    
    if (type(img_idx) == int or type(img_idx) == np.int_):
        if img_idx < 0:
            img_idx = np.random.randint(0, len(dataset))
    else:
        # replace any negative indices with random indices, with no replacement
        img_idx = np.array(img_idx)
        num_negatives = np.sum(img_idx < 0)
        img_idx[img_idx < 0] = np.random.choice(len(dataset), num_negatives, replace=False)
            
    selected_images = select_img(dataset, img_idx)

    if plot:
        plot_img(selected_images.cpu(), vmin=0, vmax=1)
    
    if scale:
        selected_images = selected_images * 2 - 1
    
    return selected_images