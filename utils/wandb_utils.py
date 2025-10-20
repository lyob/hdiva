import os
import wandb
import torch
from b_models.lvae.lvae3 import LadderVAE
from b_models.vae.vae3 import VAE
from c_training.lvae_lightning import Lightning_Model as Lightning_Model_LVAE
from c_training.lvae_config import LVAE_Training_Config
from c_training.vae_lightning import Lightning_Model as Lightning_Model_VAE
from c_training.vae_config import VAE_Training_Config
from utils.config_utils import make_config_from_dict

def get_artifact_name(
        model_num:int = 63,
        artifact_id:str = "latest",
        checkpoint_dir:str = f"/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints",
        entity:str = "blyo",
        project_name:str = "lvae",
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
    artifact = torch.load(model_path, map_location="cuda", weights_only=False)
    run = model_artifact.logged_by()
    # config = model_artifact.metadata
    config = run.config
    return model_path, artifact, config


def load_from_wandb(project_name:str, model_number:int, artifact_id='latest'):
    '''load model from wandb artifact'''
    if project_name == 'lvae':
        config_obj = LVAE_Training_Config
        lightning_model = Lightning_Model_LVAE
    elif project_name == 'vae':
        config_obj = VAE_Training_Config
        lightning_model = Lightning_Model_VAE
    model_path, artifact, config = get_artifact_name(model_number, artifact_id, project_name=project_name)
    cfg = make_config_from_dict(config_obj, config)
    lightning_model = lightning_model.load_from_checkpoint(model_path, config=cfg)
    model = lightning_model.model
    return model