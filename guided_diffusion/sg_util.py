import torch as th
import io

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file for a single-node single-GPU setup.
    """
    with open(path, "rb") as f:
        data = f.read()
    
    return th.load(io.BytesIO(data), **kwargs)


def dev():
    """
    Get the device 
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")
