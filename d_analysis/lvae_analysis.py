import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import os
    import sys
    import json
    import wandb
    import torch.nn as nn
    from torch.utils.data import DataLoader

    notebook_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    print(sys.path)

    from utils.wandb_utils import get_artifact_name
    from utils.config_utils import make_config_from_dict
    from utils.base_utils import to_01, to_0c

    from a_datasets.custom_dataset_classes import DiskDataset
    from a_datasets.hdisks3 import random_two_disk_dataset
    from b_models.lvae import LadderVAE
    from b_models.vae import VAE
    from c_training.lvae_lightning import Lightning_Model as Lightning_Model_LVAE
    from c_training.lvae_config import LVAE_Training_Config
    from c_training.vae_lightning import Lightning_Model as Lightning_Model_VAE
    from c_training.vae_config import VAE_Training_Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DataLoader,
        DiskDataset,
        LVAE_Training_Config,
        Lightning_Model_LVAE,
        Lightning_Model_VAE,
        VAE_Training_Config,
        device,
        get_artifact_name,
        make_config_from_dict,
        nn,
        plt,
        random_two_disk_dataset,
        to_01,
        to_0c,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load dataset""")
    return


@app.cell
def _(DataLoader, DiskDataset, random_two_disk_dataset):
    # hierarchical dataset 3
    data = random_two_disk_dataset(
        img_size=32,
        outer_radius=4,
        transition_width=2,
        d=10,
        num_imgs=5e4
    )[0]

    dataset = DiskDataset(data)
    dataloader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=False)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load model""")
    return


@app.cell
def _(
    LVAE_Training_Config,
    Lightning_Model_LVAE,
    Lightning_Model_VAE,
    VAE_Training_Config,
    device,
    get_artifact_name,
    make_config_from_dict,
):
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
        model = lightning_model.model.to(device)
        return model

    lvae = load_from_wandb('lvae', 12)
    vae = load_from_wandb('vae', 3)
    return lvae, vae


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Test models""")
    return


@app.cell
def _(data, device, to_0c):
    img_idx = 11
    test_img = to_0c(data[img_idx]).unsqueeze(1).to(device)
    return (test_img,)


@app.cell
def _(nn, plt, test_img, to_01, vae):
    def test_on_one_img(model, test_img, criterion = nn.MSELoss()):
        model.eval()
        xout = model(test_img)
        xout = xout.detach().cpu()
        xin = test_img.cpu()

        plot_test_one_img(xin, xout)

        # print(xin.min(), xout.min(), xin.max(), xout.max())
        test_loss = criterion(xin, xout).item()
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss

    def plot_test_one_img(xin, xout):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(to_01(xin).squeeze(0, 1), cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("Input Image")
        ax[0].axis("off")
        ax[1].imshow(to_01(xout).squeeze(0, 1), cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("Reconstructed Image")
        ax[1].axis("off")
        plt.show()

    test_loss = test_on_one_img(vae, test_img)
    return (test_on_one_img,)


@app.cell
def _(lvae, test_img, test_on_one_img):
    _ = test_on_one_img(lvae, test_img)
    return


@app.cell
def _(data, device, nn, plt, to_01, to_0c):
    test_imgs = to_0c(data[11:16]).to(device)

    def test_on_multiple_imgs(model, test_imgs, criterion = nn.MSELoss()):
        model.eval()
        out = model(test_imgs)
        xout = out.detach().cpu()
        xin = test_imgs.cpu()

        # print(xin.min(), xout.min(), xin.max(), xout.max())
        test_loss = criterion(xin, xout).item()
        print(f"Test Loss: {test_loss:.4f}")
        plot_test_multiple_imgs(xin, xout)
        return test_loss

    def plot_test_multiple_imgs(xin, xout):
        fig, ax = plt.subplots(2, len(xin), figsize=(13, 5))
        for i in range(len(xin)):
            ax[0, i].imshow(to_01(xin[i]).squeeze(0, 1), cmap="gray", vmin=0, vmax=1)
            ax[0, i].axis("off");

            ax[1, i].imshow(to_01(xout[i]).squeeze(0, 1), cmap="gray", vmin=0, vmax=1)
            ax[1, i].axis("off");
        plt.show()
    return test_imgs, test_on_multiple_imgs


@app.cell
def _(test_imgs, test_on_multiple_imgs, vae):
    test_on_multiple_imgs(vae, test_imgs);
    return


@app.cell
def _(lvae, test_imgs, test_on_multiple_imgs):
    test_on_multiple_imgs(lvae, test_imgs);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # receptive field sizes

    We're going to calculate the receptive field sizes for each of these models
    """
    )
    return


@app.cell
def _(nn, vae):
    def calculate_rf(model):
        encoder = model.encoder
        modules = []
        for bu_block in encoder:
            conv_block = bu_block.conv_block
            # print(conv_block)
            pre_conv = conv_block.pre_conv
            modules.append(pre_conv)
            block = conv_block.block
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    modules.append(layer)
        print('total number of modules that affect rf:', len(modules))

        # now calculate rf
        rf = 1
        j = 1
        for layer in modules:
            k, s, d = layer.kernel_size[0], layer.stride[0], layer.dilation[0]
            rf = rf + (k - 1) * j * d
            j = j * s
            print(rf)

    print(calculate_rf(vae));
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
