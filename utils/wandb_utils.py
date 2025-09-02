import os
import wandb
import torch

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