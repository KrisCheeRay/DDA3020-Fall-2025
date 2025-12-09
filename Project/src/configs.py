import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataSchema:
    unique_id_col: str = "unique_id"
    timestamp_col: str = "ds"
    target_col: str = "forward_returns"
    # Use the selected features
    feature_cols: List[str] = field(default_factory=lambda: [
        "E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"
    ])

@dataclass
class SFTConfig:
    """
    Configuration for Supervised Fine-Tuning (SFT) stage.
    Uses PatchTST as the primary backbone.
    """
    # Data params
    input_size: int = 60      # Lookback window size
    horizon: int = 1          # Prediction horizon (predict next step return)
    
    # PatchTST specific
    patch_len: int = 16
    d_model: int = 128         # Reduced from 256 to prevent overfitting
    n_heads: int = 4          # Reduced from 8
    n_layers: int = 3
    dropout: float = 0.2
    
    # Training params
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int = 1000     # Training steps (fixed, since early stopping is disabled)
    patience: int = 0         # Disable Early Stopping (0 > 0 is False)
    freq: int = 1             # Integer frequency for date_id steps
    revin: bool = True        # Enable Reverse Instance Normalization

@dataclass
class ArtifactPaths:
    models_dir: str = "models"
    scaler_path: str = "models/scaler.pkl"
    features_path: str = "models/features.json"
    # nf.save() creates a directory. We point to that directory.
    patchtst_save_dir: str = "models/patchtst_checkpoints" 
    nhits_save_dir: str = "models/nhits_checkpoints"
    # We keep these for compatibility or if we need explicit config access, 
    # but nf.save stores config internally.
    ensemble_config: str = "models/ensemble.json"
    policy_head_weights: str = "models/policy_head.pt"
