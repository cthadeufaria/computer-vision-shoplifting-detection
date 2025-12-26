"""
Configuration utilities for loading and managing YAML configs.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve relative paths
    config = _resolve_paths(config, config_path.parent)

    return config


def _resolve_paths(config: Dict, base_dir: Path) -> Dict:
    """
    Resolve relative paths in config to absolute paths.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths

    Returns:
        dict: Config with resolved paths
    """
    if 'data' in config and 'data_dir' in config['data']:
        data_dir = config['data']['data_dir']
        if not os.path.isabs(data_dir):
            config['data']['data_dir'] = str((base_dir / data_dir).resolve())

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        dict: Merged configuration
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        dict: Default configuration dictionary
    """
    return {
        'model': {
            'in_channels': 2,
            'num_keypoints': 17,
            'seq_len': 24,
            'num_tokens': 2,
            'gcae': {
                'hidden_channels': 64,
                'latent_channels': 8,
                'num_layers': 4,
                'dropout': 0.1
            },
            'transformer': {
                'input_dim': 136,
                'd_model': 144,
                'num_heads': 12,
                'num_layers': 4,
                'dim_feedforward': 512,
                'dropout': 0.1
            }
        },
        'training': {
            'device': 'auto',
            'stage1': {
                'epochs': 50,
                'learning_rate': 5e-5,
                'weight_decay': 1e-4
            },
            'stage2': {
                'epochs': 100,
                'learning_rate': 5e-5,
                'weight_decay': 1e-4
            },
            'batch_size': 32,
            'gradient_accumulation': 4,
            'grad_clip': 1.0,
            'scheduler': {
                'type': 'cosine_warmup',
                'warmup_epochs': 5,
                'min_lr': 1e-6
            },
            'early_stopping': {
                'enabled': True,
                'patience': 20,
                'min_delta': 0.001
            }
        },
        'data': {
            'data_dir': '../shopformer/data/PoseLift',
            'stride': 12,
            'normalize': True,
            'augmentation': {
                'enabled': True,
                'flip_prob': 0.5,
                'jitter_std': 0.02,
                'scale_range': [0.9, 1.1],
                'rotation_range': 10.0
            }
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has all required fields.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if valid

    Raises:
        ValueError: If config is invalid
    """
    required_keys = ['model', 'training', 'data']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Check model config
    model_keys = ['in_channels', 'num_keypoints', 'seq_len', 'num_tokens', 'gcae', 'transformer']
    for key in model_keys:
        if key not in config['model']:
            raise ValueError(f"Missing model config key: {key}")

    # Check transformer config
    transformer_keys = ['input_dim', 'd_model', 'num_heads', 'num_layers', 'dim_feedforward']
    for key in transformer_keys:
        if key not in config['model']['transformer']:
            raise ValueError(f"Missing transformer config key: {key}")

    # Validate d_model is divisible by num_heads
    d_model = config['model']['transformer']['d_model']
    num_heads = config['model']['transformer']['num_heads']
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    return True
