import pickle
import torch
from pathlib import Path

def load_pkl(fp):
    if isinstance(fp, (str, Path)):
        with open(fp, 'rb') as f:
            return pickle.load(f)
    else:
        return pickle.load(fp)


def to_device(lst, dev):
    if isinstance(lst, torch.Tensor):
        return lst.to(dev)
    else:
        if isinstance(lst[0], torch.Tensor):
            return [v.to(dev) for v in lst]
        else:
            return torch.as_tensor(lst, device=dev)