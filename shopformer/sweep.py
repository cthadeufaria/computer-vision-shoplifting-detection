"""
Shopformer Hyperparameter Sweep

Systematically test different hyperparameter configurations to find optimal settings.

Usage:
    python sweep.py --data_dir ./data/PoseLift --output_dir ./sweep_results
    python sweep.py --use_synthetic --quick  # Quick test run
"""

import os
import argparse
import json
import itertools
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np


# Define hyperparameter search space
SEARCH_SPACE = {
    # Model architecture
    'hidden_channels': [64, 128],
    'latent_channels': [8, 16],
    'transformer_heads': [2, 4],
    'transformer_layers': [2, 3],
    'transformer_ff_dim': [64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'num_tokens': [2, 4],

    # Training
    'lr': [1e-4, 5e-5, 1e-5],
    'batch_size': [16, 32],
    'weight_decay': [1e-4, 1e-5],
    'scheduler': ['cosine', 'plateau'],

    # Augmentation
    'jitter_std': [0.01, 0.02, 0.03],
}

# Quick search space for testing
QUICK_SEARCH_SPACE = {
    'hidden_channels': [64, 128],
    'latent_channels': [8, 16],
    'transformer_layers': [2, 3],
    'dropout': [0.1, 0.2],
    'lr': [1e-4, 5e-5],
}

# Recommended configurations to try first
RECOMMENDED_CONFIGS = [
    {
        'name': 'baseline',
        'hidden_channels': 64,
        'latent_channels': 8,
        'transformer_heads': 2,
        'transformer_layers': 2,
        'transformer_ff_dim': 64,
        'dropout': 0.1,
        'lr': 5e-5,
        'batch_size': 32,
        'stage1_epochs': 30,
        'stage2_epochs': 50,
    },
    {
        'name': 'deeper_wider',
        'hidden_channels': 128,
        'latent_channels': 16,
        'transformer_heads': 4,
        'transformer_layers': 3,
        'transformer_ff_dim': 128,
        'dropout': 0.2,
        'lr': 1e-4,
        'batch_size': 32,
        'stage1_epochs': 40,
        'stage2_epochs': 60,
    },
    {
        'name': 'high_regularization',
        'hidden_channels': 64,
        'latent_channels': 8,
        'transformer_heads': 2,
        'transformer_layers': 2,
        'transformer_ff_dim': 64,
        'dropout': 0.3,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'batch_size': 16,
        'stage1_epochs': 30,
        'stage2_epochs': 50,
    },
    {
        'name': 'more_tokens',
        'hidden_channels': 64,
        'latent_channels': 16,
        'transformer_heads': 4,
        'transformer_layers': 2,
        'transformer_ff_dim': 128,
        'dropout': 0.2,
        'num_tokens': 4,
        'lr': 5e-5,
        'batch_size': 32,
        'stage1_epochs': 30,
        'stage2_epochs': 50,
    },
    {
        'name': 'aggressive_augmentation',
        'hidden_channels': 128,
        'latent_channels': 8,
        'transformer_heads': 2,
        'transformer_layers': 2,
        'transformer_ff_dim': 64,
        'dropout': 0.2,
        'lr': 1e-4,
        'batch_size': 32,
        'jitter_std': 0.03,
        'scale_range': [0.9, 1.1],
        'rotation_range': 0.1,
        'temporal_dropout': 0.1,
        'stage1_epochs': 40,
        'stage2_epochs': 60,
    },
]


def generate_grid_configs(search_space: Dict, max_configs: int = None) -> List[Dict]:
    """Generate all combinations from search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    if max_configs and len(configs) > max_configs:
        # Random sample
        np.random.shuffle(configs)
        configs = configs[:max_configs]

    return configs


def generate_random_configs(search_space: Dict, n_configs: int = 20) -> List[Dict]:
    """Generate random configurations from search space."""
    configs = []
    for i in range(n_configs):
        config = {}
        for key, values in search_space.items():
            config[key] = np.random.choice(values)
        configs.append(config)
    return configs


def run_training(config: Dict, data_dir: str, output_dir: str,
                 use_synthetic: bool = False, device: str = 'auto') -> Dict:
    """Run training with given configuration."""
    # Build command
    cmd = [
        sys.executable, 'train.py',
        '--output_dir', output_dir,
        '--device', device,
    ]

    if use_synthetic:
        cmd.append('--use_synthetic')
    else:
        cmd.extend(['--data_dir', data_dir])

    # Add config parameters
    for key, value in config.items():
        if key == 'name':
            continue
        if key == 'scale_range':
            cmd.extend([f'--{key}', str(value[0]), str(value[1])])
        elif isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"\nRunning: {' '.join(cmd)}")

    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        # Parse results
        if result.returncode == 0:
            # Load results from checkpoint
            config_path = Path(output_dir) / 'config.json'
            best_model_path = Path(output_dir) / 'best_model.pt'

            if best_model_path.exists():
                import torch
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                return {
                    'success': True,
                    'best_auc': checkpoint.get('best_auc', 0),
                    'best_epoch': checkpoint.get('epoch', 0),
                    'metrics': checkpoint.get('metrics', {}),
                    'stdout': result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
                }
        return {
            'success': False,
            'error': result.stderr[-1000:] if result.stderr else 'Unknown error',
            'stdout': result.stdout[-1000:] if result.stdout else ''
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Training timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_sweep(
    configs: List[Dict],
    data_dir: str,
    output_base: str,
    use_synthetic: bool = False,
    device: str = 'auto'
) -> List[Dict]:
    """Run sweep over all configurations."""
    results = []

    for i, config in enumerate(configs):
        config_name = config.get('name', f'config_{i:03d}')
        print(f"\n{'='*60}")
        print(f"Running configuration {i+1}/{len(configs)}: {config_name}")
        print(f"{'='*60}")
        print(f"Config: {json.dumps(config, indent=2)}")

        # Create output directory for this config
        output_dir = Path(output_base) / config_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / 'sweep_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Run training
        result = run_training(
            config, data_dir, str(output_dir),
            use_synthetic=use_synthetic, device=device
        )

        result['config'] = config
        result['config_name'] = config_name
        results.append(result)

        # Print result
        if result['success']:
            print(f"\nResult: AUC-ROC = {result['best_auc']:.4f} at epoch {result['best_epoch']}")
        else:
            print(f"\nFailed: {result.get('error', 'Unknown error')}")

        # Save intermediate results
        with open(Path(output_base) / 'sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze sweep results and find best configuration."""
    successful = [r for r in results if r['success']]

    if not successful:
        return {'error': 'No successful runs'}

    # Sort by AUC
    successful.sort(key=lambda x: x.get('best_auc', 0), reverse=True)

    analysis = {
        'total_runs': len(results),
        'successful_runs': len(successful),
        'failed_runs': len(results) - len(successful),
        'best_config': successful[0]['config'],
        'best_auc': successful[0]['best_auc'],
        'best_config_name': successful[0]['config_name'],
        'top_5': [
            {
                'name': r['config_name'],
                'auc': r['best_auc'],
                'config': r['config']
            }
            for r in successful[:5]
        ],
        'all_results': [
            {
                'name': r['config_name'],
                'auc': r.get('best_auc', 0),
                'success': r['success']
            }
            for r in results
        ]
    }

    # Parameter importance analysis
    if len(successful) > 3:
        param_scores = {}
        for param in successful[0]['config'].keys():
            if param == 'name':
                continue
            param_scores[param] = {}
            for r in successful:
                val = str(r['config'].get(param))
                if val not in param_scores[param]:
                    param_scores[param][val] = []
                param_scores[param][val].append(r['best_auc'])

        # Average AUC per parameter value
        param_analysis = {}
        for param, val_scores in param_scores.items():
            param_analysis[param] = {
                val: {'mean_auc': np.mean(scores), 'count': len(scores)}
                for val, scores in val_scores.items()
            }
        analysis['parameter_analysis'] = param_analysis

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Shopformer Hyperparameter Sweep")
    parser.add_argument('--data_dir', type=str, default='./data/PoseLift')
    parser.add_argument('--output_dir', type=str, default='./sweep_results')
    parser.add_argument('--use_synthetic', action='store_true')
    parser.add_argument('--device', type=str, default='auto')

    # Sweep mode
    parser.add_argument('--mode', type=str, default='recommended',
                        choices=['recommended', 'grid', 'random', 'quick'])
    parser.add_argument('--n_random', type=int, default=20,
                        help='Number of random configs (for random mode)')
    parser.add_argument('--max_grid', type=int, default=50,
                        help='Maximum grid configs to try')

    # Single config test
    parser.add_argument('--single', type=str, default=None,
                        help='Run single config by name from recommended')

    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"sweep_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Shopformer Hyperparameter Sweep")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Output: {output_base}")
    print(f"Data: {'synthetic' if args.use_synthetic else args.data_dir}")

    # Generate configurations
    if args.single:
        configs = [c for c in RECOMMENDED_CONFIGS if c.get('name') == args.single]
        if not configs:
            print(f"Config '{args.single}' not found. Available: {[c['name'] for c in RECOMMENDED_CONFIGS]}")
            return
    elif args.mode == 'recommended':
        configs = RECOMMENDED_CONFIGS
    elif args.mode == 'quick':
        configs = generate_random_configs(QUICK_SEARCH_SPACE, n_configs=5)
        # Add reduced epochs for quick testing
        for c in configs:
            c['stage1_epochs'] = 10
            c['stage2_epochs'] = 20
    elif args.mode == 'grid':
        configs = generate_grid_configs(SEARCH_SPACE, max_configs=args.max_grid)
    elif args.mode == 'random':
        configs = generate_random_configs(SEARCH_SPACE, n_configs=args.n_random)
    else:
        configs = RECOMMENDED_CONFIGS

    print(f"Total configurations to test: {len(configs)}")

    # Save sweep configuration
    sweep_info = {
        'timestamp': timestamp,
        'mode': args.mode,
        'n_configs': len(configs),
        'data_dir': args.data_dir,
        'use_synthetic': args.use_synthetic,
        'configs': configs
    }
    with open(output_base / 'sweep_info.json', 'w') as f:
        json.dump(sweep_info, f, indent=2, default=str)

    # Run sweep
    results = run_sweep(
        configs,
        args.data_dir,
        str(output_base),
        use_synthetic=args.use_synthetic,
        device=args.device
    )

    # Analyze results
    analysis = analyze_results(results)

    # Save analysis
    with open(output_base / 'analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    print(f"Total runs: {analysis.get('total_runs', 0)}")
    print(f"Successful: {analysis.get('successful_runs', 0)}")
    print(f"Failed: {analysis.get('failed_runs', 0)}")

    if 'best_config' in analysis:
        print(f"\nBest Configuration: {analysis['best_config_name']}")
        print(f"Best AUC-ROC: {analysis['best_auc']:.4f}")
        print(f"\nConfig:")
        print(json.dumps(analysis['best_config'], indent=2))

        print("\nTop 5 Configurations:")
        for i, top in enumerate(analysis.get('top_5', []), 1):
            print(f"  {i}. {top['name']}: AUC = {top['auc']:.4f}")

    print(f"\nResults saved to: {output_base}")


if __name__ == '__main__':
    main()
