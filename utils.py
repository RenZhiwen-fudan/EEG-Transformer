import numpy as np
import torch
import random
import os

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_class_balance(labels):
    """Check and report class balance"""
    class_counts = np.bincount(labels)
    print("\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples ({count/len(labels)*100:.2f}%)")
    return class_counts