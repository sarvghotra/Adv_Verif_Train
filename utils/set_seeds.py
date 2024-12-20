import random
import numpy as np
import torch


def set_seeds(seed=42):
    # Set the Python seed for the random module
    random.seed(seed)
    
    # Set the seed for numpy
    np.random.seed(seed)
    
    # Set the seed for PyTorch on CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    # Ensure deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
