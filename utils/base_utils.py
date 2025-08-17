import numpy as np
import torch
from typing import Union

def t2n(t:torch.Tensor)->np.ndarray:    
    return t.detach().cpu().numpy()

def send_to_device(flag:bool,obj:Union[torch.nn.Module,torch.Tensor]):
    if flag and torch.cuda.is_available():
        if isinstance(obj, torch.nn.Module):
            return obj.cuda()
        elif isinstance(obj, torch.Tensor):
            return obj.to('cuda')
    return obj

def plant(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def broadcast(x, like):
    return x.view(-1, *((1,) * (len(like.shape) - 1)))


def to_01(img):
    # img = img * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    img = (img + 1) / 2.0
    return img

def to_0c(img):
    img = img * 2 - 1  # Map from (0, 1) to (-1, 1)
    return img

