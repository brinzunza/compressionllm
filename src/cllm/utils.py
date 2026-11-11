"""Small shared helpers: devices, seeding, timing, memory, logging."""
import contextlib
import os
import platform
import random
import time

import numpy as np
import torch


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get_device(pref="auto"):
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def peak_memory_mb(device):
    """Peak accelerator memory since reset (CUDA), else current allocator/process memory."""
    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if d.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return rss_mb()


@contextlib.contextmanager
def exact_fp32():
    """Disable TF32 so logits are as reproducible as possible between encoder and decoder."""
    a, b = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = a, b


def hardware_info():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 1), cuda=torch.version.cuda)
    elif torch.backends.mps.is_available():
        info["gpu"] = "Apple MPS"
    return info
