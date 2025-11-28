import os
import wandb
import torch
from dataclasses import fields

def make_config_from_dict(cls, d: dict):
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in field_names}
    return cls(**filtered)

