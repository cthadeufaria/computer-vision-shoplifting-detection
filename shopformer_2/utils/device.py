"""
Device utilities for MPS/CUDA/CPU selection.

Optimized for Apple Silicon with MPS (Metal Performance Shaders) backend.
"""

import os
import torch


def get_device(preference: str = 'auto') -> torch.device:
    """
    Get the best available device for PyTorch.

    Priority: MPS > CUDA > CPU

    Args:
        preference: 'auto', 'mps', 'cuda', or 'cpu'

    Returns:
        torch.device: Selected device
    """
    if preference == 'cpu':
        print("Using CPU (as requested)")
        return torch.device('cpu')

    if preference == 'mps' or preference == 'auto':
        if torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            return torch.device('mps')
        elif preference == 'mps':
            print("Warning: MPS requested but not available, falling back to CPU")

    if preference == 'cuda' or preference == 'auto':
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA: {device_name}")
            return torch.device('cuda')
        elif preference == 'cuda':
            print("Warning: CUDA requested but not available, falling back to CPU")

    print("Using CPU")
    return torch.device('cpu')


def check_mps_availability() -> bool:
    """
    Check if MPS backend is available and properly configured.

    Returns:
        bool: True if MPS is available
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available: PyTorch not built with MPS support")
        else:
            print("MPS not available: macOS < 12.3 or no Apple Silicon")
        return False
    return True


def setup_mps_environment():
    """
    Setup environment variables for optimal MPS performance.

    Call this at the start of training.
    """
    # Enable fallback to CPU for unsupported operations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Optionally disable MPS memory warnings
    # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def clear_mps_cache():
    """
    Clear MPS memory cache.

    Useful when running out of memory during training.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def print_device_info():
    """Print detailed device information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")


def move_to_device(data, device: torch.device):
    """
    Recursively move data to device.

    Handles tensors, lists, tuples, and dicts.

    Args:
        data: Data to move (tensor, list, tuple, or dict)
        device: Target device

    Returns:
        Data on the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(d, device) for d in data)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    return data
