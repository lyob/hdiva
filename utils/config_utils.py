import os
import wandb
import torch
from dataclasses import fields
from b_models.base_modules.unet import UNet
from b_models.base_modules.half_unet import Half_UNet
from b_models.diva.diva_module import DiVA

def make_config_from_dict(cls, d: dict):
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in field_names}
    return cls(**filtered)


def init_diva_model(config):
    denoiser = UNet(config)
    infnet = Half_UNet(config)
    model = DiVA(denoiser, infnet, config.noise_schedule, config.sigma_minmax, config.timestep_dist, config.num_timesteps, config.reduction)
    return model


def get_artifact_name(
        project_name:str = "hq_autoprior",
        model_num:int = 63,
        artifact_id:str = "latest",
        checkpoint_dir:str = f"/mnt/home/blyo1/diva/celebahq/lightning_checkpoints",
        entity:str = "blyo",
        map_location:str = "cuda"
    ):
    '''find the artifact name, e.g. "model-kt6xg4ur'''
    if isinstance(model_num, int):
        model_num = str(model_num)
    
    for dir in os.listdir(checkpoint_dir):
        if dir.startswith(model_num):
            checkpoint_dir = f"{checkpoint_dir}/{dir}"
            break

    # get the folder after /
    artifact_name = checkpoint_dir.split("/")[-1]
    artifact_name = artifact_name.split("-")[-1]
    artifact_name = "model-" + artifact_name
    # print(artifact_name)

    # download the artifact and config
    api = wandb.Api()
    model_artifact = api.artifact(f'{entity}/{project_name}/{artifact_name}:{artifact_id}')
    model_dir = model_artifact.download()
    model_path = f'{model_dir}/model.ckpt'
    artifact = torch.load(model_path, map_location, weights_only=False)
    config = model_artifact.metadata

    source_run = model_artifact.logged_by()
    run_config = source_run.config
    return model_path, artifact, config, run_config


def load_pretrained_module_from_wandb(wandb_config, new_model, module_name='denoiser'):
    """Load specific module weights from a wandb checkpoint."""
    try:
        # Download checkpoint from wandb and load the checkpoint
        checkpoint = get_artifact_name(**wandb_config, map_location='cpu')[1]
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Extract module-specific weights
        module_state_dict = {}
        prefix = f"model.{module_name}."
        
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, '')
                module_state_dict[new_key] = value
        
        if module_state_dict:
            # Get the target module
            target_module = getattr(new_model, module_name)
            
            # Load weights
            missing_keys, unexpected_keys = target_module.load_state_dict(
                module_state_dict, strict=False
            )

            print(f"Loaded {module_name} weights from wandb artifact_id {wandb_config['artifact_id']}")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"No {module_name} weights found in wandb checkpoint")
        
        return new_model
            
    except Exception as e:
        print(f"Error loading from wandb: {e}")
