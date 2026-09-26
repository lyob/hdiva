import argparse
import os
from typing import Any, Tuple

import numpy as np
import torch

import wandb


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


def init_refactored_autoprior_using_args(args:argparse.Namespace, device:torch.device, send_to_device:bool=True):
    '''from an args object, load the autoprior model'''
    from b_models.autoprior.celeba_autoprior_refactor import (
        AutoPrior_args, AutoPrior_Inference)
    from b_models.autoprior.celeba_denoisers_t_as_input import \
        UNet as UNet_t_as_input
    from b_models.autoprior.celeba_encoders import BF_CNN_RF as BF_CNN_RF_enc
    from b_models.autoprior.celeba_encoders import (HalfUNet,
                                                    VariableConvEncoder)
    
    if args.encoder_model == 'half_unet':
        encoder = HalfUNet(args)
    elif args.encoder_model == 'simple_convnet':
        encoder = VariableConvEncoder(args)
    elif args.encoder_model == 'bf_cnn_rf':
        encoder = BF_CNN_RF_enc(args)
    else:
        raise ValueError('Encoder model not recognized. Should be either half_unet or simple_convnet or bf_cnn_rf')
    
    if args.denoiser_model == 'unet':
        denoiser = UNet_t_as_input(args)
    else:
        raise ValueError('Denoiser model not recognized. Should be unet or bf_cnn_rf')
    
    autoprior = AutoPrior_Inference(
        encoder, 
        denoiser,
        args, 
        device
    )
    
    if send_to_device:
        autoprior = autoprior.to(device)
    return autoprior

def load_model_args_from_artifact(entity, project, model_artifact_name, model_alias):
    api = wandb.Api()
    model_artifact = api.artifact(f'{entity}/{project}/{model_artifact_name}:{model_alias}')
    model_dir = model_artifact.download()
    model_path = f'{model_dir}/final_model_weights.pt'
    config = model_artifact.metadata
    # # config['timesteps'] = 1000
    args = argparse.Namespace(**config)
    return config, args, model_path



def load_autoprior_weights(args, model_path, device):
    '''load the weights of the autoprior model'''
    autoprior = init_refactored_autoprior_using_args(args, device)

    # load weights
    state_dict = torch.load(model_path, map_location=device)

    if 'train_only_denoiser' in args and args.train_only_denoiser:
        print('loading only denoiser weights')
        # load the denoiser weights from state dict, ignoring the encoder keys
        encoder_keys = [key for key in state_dict.keys() if 'encoder' in key]
        for key in encoder_keys:
            state_dict.pop(key)
        # print(state_dict.keys())
        
    # if state_dict has extra keys ignore them
    state_dict = {k: v for k, v in state_dict.items() if k in autoprior.state_dict()}
    autoprior.load_state_dict(state_dict)
    
    return autoprior


def load_trained_sami_autoprior(
    model_num:int,
    checkpoint:str="last",
    from_wandb:bool=False,
    artifact_id:str="latest",
    project_name:str="sami_autoprior_celeba_64",
    device:torch.device=torch.device('cpu'),
    checkpoint_path:str|None=None,
):
    '''load a model trained with c_training/sami_autoprior_train.py into AutoPrior_Inference, so the
    autoprior sampling methods (conditional_sample, conditional_sample_from_z, ...) can be used with it.

    model_num: the leading number of the run dir, e.g. 2 for "2-eager-eon-2-u02iqu2g"
    checkpoint: "last" or a zero-padded epoch, e.g. "0049" (local checkpoints only)
    from_wandb: download the checkpoint artifact logged by WandbArtifactCallback instead of using the local file
    artifact_id: wandb artifact version, e.g. "latest" or "v3" (only used if from_wandb)
    checkpoint_path: load this .ckpt file directly (e.g. a copy pinned outside the rotating checkpoint dir);
        overrides model_num/checkpoint/from_wandb
    '''
    from dataclasses import asdict

    from b_models.autoprior.sami_autoprior import get_autoprior_weights_path
    from b_models.configs.sami_autoprior_config import \
        SAMI_AutoPrior_CelebA_64_Training_Config
    from utils.training import get_checkpoint_dir
    from utils.wandb_utils import get_artifact

    if checkpoint_path is not None:
        model_path = checkpoint_path
    elif from_wandb:
        model_path = get_artifact(model_num, artifact_id, project_name=project_name, map_location='cpu')[0]
    else:
        model_path = get_checkpoint_dir(model_num, project_name, epoch=checkpoint)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = SAMI_AutoPrior_CelebA_64_Training_Config.from_dict(ckpt['hyper_parameters'])

    # the autoprior classes read a few fields under their original names
    args = argparse.Namespace(
        **asdict(config),
        timesteps=config.num_timesteps,
        noise_dist=config.noise_schedule,
        denoiser_target=config.parameterization,
        weighted_MSE=config.weighted_mse,
        rescale=True,  # inputs in [0, 1], rescaled to [-1, 1] internally, as during training
        kl_reduction='sum',
        log_posterior_method='z_given_xt',
        train_only_denoiser=False,
        encoder_model='half_unet',
        denoiser_model='unet',
    )
    autoprior = init_refactored_autoprior_using_args(args, device, send_to_device=False)

    state_dict = ckpt['state_dict']
    denoiser_sd = {k.removeprefix('model.denoiser.'): v for k, v in state_dict.items() if k.startswith('model.denoiser.')}
    encoder_sd = {k.removeprefix('model.infnet.'): v for k, v in state_dict.items() if k.startswith('model.infnet.')}
    if not denoiser_sd:
        # train_infnet_only runs drop the (frozen) denoiser from their checkpoints, so use the pretrained one
        pretrained_sd = torch.load(get_autoprior_weights_path(config), map_location='cpu')
        denoiser_sd = {k.removeprefix('denoiser.'): v for k, v in pretrained_sd.items() if k.startswith('denoiser.')}
    autoprior.denoiser.load_state_dict(denoiser_sd, strict=True)
    autoprior.encoder.load_state_dict(encoder_sd, strict=True)
    print(f'loaded {model_path} (epoch {ckpt.get("epoch")})')

    return autoprior.to(device).eval(), config


def get_celeba_dataset(dataset_resolution:int, training_set_size:int, from_tmp=False, base_path = './a_datasets/celeba'):
    # dataset_size = int(dataset_name.split('-')[-1])
    # dataset_shape = int(dataset_name.split('-')[-2].split('x')[0])
    # data_path = os.path.join('data/celeba', f'train{dataset_shape}x{dataset_shape}_no_repeats.pt')
    if from_tmp:
        base_path = '/tmp'
    else:
        # base_path = os.path.join(base_path, './datasets/celeba')
        base_path = os.path.join(base_path)
    
    data_path = os.path.join(base_path, f'attribute_images_{dataset_resolution}x{dataset_resolution}.pt')
    dataset = torch.load(data_path, map_location='cpu')
    
    train_subset = dataset[:training_set_size]
    test_subset = dataset[training_set_size:]

    print('training set size: ', train_subset.shape[0] )
    print('test set size: ', test_subset.shape[0] )
    
    return train_subset, test_subset
