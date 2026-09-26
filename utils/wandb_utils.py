import os
import time

import torch
import wandb

from b_models.lvae.lvae3 import LadderVAE
from b_models.lvae.lvae_lightning import Lightning_Model as Lightning_Model_LVAE
from b_models.vae.vae3 import VAE
from b_models.configs.lvae_config import LVAE_Training_Config

# from b_models.configs.vae_config import VAE_Training_Config
from utils.config_utils import make_config_from_dict


def get_wandb_run_name(
    model_num: int = 63,
    checkpoint_dir: str = f"/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints/lvae",
):
    """find the artifact name, e.g. "model-kt6xg4ur"""
    print(checkpoint_dir)
    model_num = str(model_num)

    # run dirs are named "<model_num>-<run name>-<run id>", so match the leading number exactly
    # (a prefix match would let model 1 pick up 10, 11, ...)
    matches = [d for d in os.listdir(checkpoint_dir) if d.split("-")[0] == model_num]
    if len(matches) != 1:
        raise ValueError(f"expected one run dir for model {model_num} in {checkpoint_dir}, found {matches}")

    wandb_run_name = matches[0].split("-")[-1]
    # wandb_run_name = "-".join(wandb_run_name)
    print(f"wandb run name: {wandb_run_name}")
    return wandb_run_name


def download_artifact_rank0(model_artifact, root: str, timeout: int = 900, poll_every: int = 2):
    """Download a wandb artifact once, on global rank 0 only.

    Every rank calls this, but only rank 0 writes to `root`; the others wait for a
    sentinel file that rank 0 drops after the download returns. Without this, all
    ranks race to write the same .ckpt and the losers torch.load a half-written
    file ("PytorchStreamReader ... archive is corrupted").

    This runs during model __init__, before the process group exists, so it uses a
    file sentinel rather than torch.distributed.barrier().
    """
    rank = int(os.environ.get("RANK", 0))
    sentinel = os.path.join(root, ".download_complete")

    if rank == 0:
        # Clear any sentinel from a previous run so other ranks can't read it as
        # a go-ahead while this download is still in flight.
        if os.path.exists(sentinel):
            os.remove(sentinel)
        model_dir = model_artifact.download(root=root)
        with open(sentinel, "w") as f:
            f.write(str(time.time()))
        return model_dir

    deadline = time.time() + timeout
    while not os.path.exists(sentinel):
        if time.time() > deadline:
            raise TimeoutError(
                f"rank {rank} waited {timeout}s for rank 0 to download the artifact into {root}"
            )
        time.sleep(poll_every)
    return root


def get_artifact(
    model_num: int = 63,
    artifact_id: str = "latest",
    checkpoint_dir: str = f"/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints",
    entity: str = "blyo",
    project_name: str = "lvae",
    map_location: str = "cuda",
):
    """find the artifact name, e.g. "model-kt6xg4ur"""
    checkpoint_dir = f"{checkpoint_dir}/{project_name}"
    artifact_name = get_wandb_run_name(model_num=model_num, checkpoint_dir=checkpoint_dir)
    artifact_name = "model-" + artifact_name
    # print(artifact_name)

    # download the artifact and config
    api = wandb.Api()
    model_artifact = api.artifact(f"{entity}/{project_name}/{artifact_name}:{artifact_id}")
    model_dir = download_artifact_rank0(
        model_artifact, root=f"/mnt/home/blyo1/hdiva/artifacts/{artifact_name}:{artifact_id}"
    )
    model_path = f"{model_dir}/model.ckpt"
    # if this model_path does not exist, look for any .ckpt file in the directory
    if not os.path.exists(model_path):
        for file in os.listdir(model_dir):
            if file.endswith(".ckpt"):
                model_path = os.path.join(model_dir, file)
                break
    artifact = torch.load(model_path, map_location, weights_only=False)
    config = model_artifact.metadata

    source_run = model_artifact.logged_by()
    run_config = source_run.config
    return model_path, artifact, config, run_config


def load_from_wandb(project_name: str, model_number: int, artifact_id="latest"):
    """load model from wandb artifact"""
    if project_name == "lvae":
        config_obj = LVAE_Training_Config
        lightning_model = Lightning_Model_LVAE
    elif project_name == "vae":
        # Lazy import to avoid circular import
        from b_models.vae.vae_lightning import VAE_Lightning
        from b_models.configs.vae_config import SelectConfig

        config_obj = SelectConfig
        # config_obj = VAE_Training_Config
        lightning_model = VAE_Lightning
    model_path, artifact, config = get_artifact(model_number, artifact_id, project_name=project_name)
    cfg = make_config_from_dict(config_obj, config)
    lightning_model = lightning_model.load_from_checkpoint(model_path, config=cfg)
    model = lightning_model.model
    return model


def load_pretrained_module_from_wandb(wandb_config, new_model, module_name="denoiser"):
    """Load specific module weights from a wandb checkpoint."""
    try:
        # Download checkpoint from wandb and load the checkpoint
        checkpoint = get_artifact(**wandb_config, map_location="cpu")[1]

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Extract module-specific weights
        module_state_dict = {}
        prefix = f"model.{module_name}."

        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                module_state_dict[new_key] = value

        if module_state_dict:
            # Get the target module
            target_module = getattr(new_model, module_name)

            # Load weights
            missing_keys, unexpected_keys = target_module.load_state_dict(module_state_dict, strict=False)

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
        raise
