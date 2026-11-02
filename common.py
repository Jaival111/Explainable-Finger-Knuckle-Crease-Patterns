"""Small shared helpers: seeding, device, checkpoints, tensor conversion."""
import random
from pathlib import Path

import numpy as np
import torch

import config


def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, path, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(model, path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train the corresponding stage first.")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})


def images_to_tensor(images):
    """List of (H, W) float32 arrays in [0, 1] -> (B, 1, H, W) tensor."""
    return torch.from_numpy(np.stack(images)[:, None].astype(np.float32))


def gather_rows(x, idx):
    """x: (B, N, C), idx: (B, c) -> (B, c, C)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
