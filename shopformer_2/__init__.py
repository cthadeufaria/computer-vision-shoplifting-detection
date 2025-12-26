"""
Shopformer_2: Paper-aligned Shopformer implementation

This implementation matches the paper specifications:
- 12 attention heads, 4 transformer layers
- GCAE tokenizer with 4 ST-GCN layers
- Two-stage training: GCAE then Transformer
- MPS (Apple Silicon) support
"""

__version__ = "2.0.0"
