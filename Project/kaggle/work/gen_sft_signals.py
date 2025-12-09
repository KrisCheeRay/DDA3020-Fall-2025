import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import TimeSeriesSplit
from neuralforecast import NeuralForecast

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore
from kaggle.work.sft_patchTST import build_model as build_patchtst

# Try to import NHITS builder, skip if not available
try:
    from kaggle.work.sft_nhits import build_model as build_nhits
except ImportError:
    build_nhits = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ensemble_config(artifacts: ArtifactPaths) -> Dict:
    """Read ensemble config, return default if not exists"""
    if Path(artifacts.ensemble_config).exists():
        with open(artifacts.ensemble_config, "r") as f:
            return json.load(f)
    return {"type": "fixed", "weights": {"patchtst": 1.0, "nhits": 0.0}}

def compute_recent_vol(df: pl.DataFrame, window: int = 20) -> np.ndarray:
    """
    Compute recent volatility for dynamic weighting.
    """
    # Simple rolling std of y
    vol = df.select(pl.col("y").rolling_std(window_size=window).fill_null(0.0)).to_series().to_numpy()
    return vol

def combine_predictions(
    preds_patch: np.ndarray, 
    preds_nhits: Optional[np.ndarray], 
    vol: np.ndarray, 
    config: Dict
) -> np.ndarray:
    """
    Combine predictions based on ensemble.json logic
    """
    if preds_nhits is None:
        return preds_patch

    # 1. Fixed Weights
    if config.get("type", "fixed") == "fixed":
        weights = config.get("weights", {"patchtst": 1.0, "nhits": 0.0})
        w_p = weights.get("patchtst", 1.0)
        w_n = weights.get("nhits", 0.0)
        return w_p * preds_patch + w_n * preds_nhits

    # 2. Regime-based Dynamic Weights
    regime = config.get("regime_vol", {})
    if regime.get("enabled", False):
        thresholds = regime.get("thresholds", [0.01, 0.03])
        w_low = regime["weights"]["low"]
        w_mid = regime["weights"]["mid"]
        w_high = regime["weights"]["high"]
        
        final_preds = np.zeros_like(preds_patch)
        
        mask_low = vol < thresholds[0]
        mask_mid = (vol >= thresholds[0]) & (vol < thresholds[1])
        mask_high = vol >= thresholds[1]
        
        final_preds[mask_low] = (
            w_low["patchtst"] * preds_patch[mask_low] + 
            w_low["nhits"] * preds_nhits[mask_low]
        )
        final_preds[mask_mid] = (
            w_mid["patchtst"] * preds_patch[mask_mid] + 
            w_mid["nhits"] * preds_nhits[mask_mid]
        )
        final_preds[mask_high] = (
            w_high["patchtst"] * preds_patch[mask_high] + 
            w_high["nhits"] * preds_nhits[mask_high]
        )
        return final_preds

    return preds_patch

def generate_oos_predictions(
    df: pl.DataFrame,
    schema: DataSchema,
    sft_cfg: SFTConfig,
    artifacts: ArtifactPaths
) -> None:
    """
    Core function: Execute K-Fold Cross Validation to generate "Rotten Apple" (OOS) data.
    """
    ensemble_cfg = get_ensemble_config(artifacts)
    logger.info(f"Loaded Ensemble Config: {ensemble_cfg}")
    
    # Global Feature Store Fit (Simplified)
    fs = FeatureStore(schema=schema)
    df_scaled = fs.fit_transform(df) # Global Scaling
    
    # Build Models
    models = []
    # PatchTST: No hist_exog (CI mode)
    patch_model = build_patchtst(sft_cfg)
    models.append(patch_model)
    
    if build_nhits and ensemble_cfg.get("weights", {}).get("nhits", 0.0) > 0:
        # NHITS: Supports hist_exog (Multivariate mode)
        # Note: sft_nhits.build_model internally uses schema.feature_cols for hist_exog
        models.append(build_nhits(sft_cfg, schema))
        
    nf = NeuralForecast(models=models, freq=sft_cfg.freq)
    
    # Calculate required backtest windows
    total_steps = df.height
    input_size = sft_cfg.input_size
    n_windows = total_steps - input_size - 1
    
    if n_windows <= 0:
        logger.warning("Data too short for cross validation.")
        return

    logger.info(f"Starting Cross Validation with n_windows={n_windows}...")
    
    # step_size=1: Predict next day every day (Rolling Window)
    # refit=False: No retraining for speed
    cv_df = nf.cross_validation(
        df=df_scaled.to_pandas(),
        n_windows=n_windows,
        step_size=1,
        refit=False 
    )
    cv_df = pl.from_pandas(cv_df)
    
    # 3. Apply Ensemble
    pred_patch = cv_df["PatchTST"].to_numpy()
    
    pred_nhits = None
    if "NHITS" in cv_df.columns:
        pred_nhits = cv_df["NHITS"].to_numpy()
        
    # Compute volatility (based on actual y)
    vol = compute_recent_vol(cv_df)
    
    final_y_hat = combine_predictions(pred_patch, pred_nhits, vol, ensemble_cfg)
    
    # 4. Save Results (Enhanced for GRPO)
    # Save both individual predictions so GRPO can learn dynamic weights
    result_df = cv_df.select([
        pl.col("unique_id"),
        pl.col("ds"),
        pl.col("y").alias("y_true"),
    ]).with_columns([
        pl.Series("patchtst_y_hat", pred_patch),
        pl.Series("vol", vol),
        pl.Series("y_hat", final_y_hat) # Keep the ensemble result as baseline
    ])
    
    if pred_nhits is not None:
        result_df = result_df.with_columns(
            pl.Series("nhits_y_hat", pred_nhits)
        )
    
    out_path = Path("models/sft_y_hat_oos.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(out_path)
    logger.info(f"Saved OOS predictions to {out_path}, shape: {result_df.shape}")

if __name__ == "__main__":
    from kaggle.work.sft_patchTST import to_long_format
    
    data_root = Path("kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
        
    train_path = data_root / "train_feature_selected.csv"
    if train_path.exists():
        print(f"Loading data from {train_path}...")
        train = pl.read_csv(train_path)
        
        # CRITICAL FIX: Shift forward_returns by 1
        if "forward_returns" in train.columns:
            train = train.with_columns(
                pl.col("forward_returns").shift(1).alias("y")
            )
        if "date_id" in train.columns:
            train = train.rename({"date_id": "ds"})
            
        # DROP NULLS
        feature_cols = ["E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"]
        original_len = len(train)
        
        # Data Cleaning Strategy:
        # 1. Count nulls in feature columns
        null_counts = train.select(
            pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
        )
        train = train.with_columns(null_counts)
        
        # 2. Filter: Keep rows with <= 4 nulls AND valid y
        train = train.filter(
            (pl.col("null_count") <= 4) & (pl.col("y").is_not_null())
        )
        
        # 3. Fill remaining nulls with 0.0
        train = train.with_columns([
            pl.col(c).fill_null(0.0) for c in feature_cols
        ])
        train = train.drop("null_count") # cleanup
        
        print(f"Data Cleaning: Dropped {original_len - len(train)} rows. Remaining: {len(train)}")

        # Add unique_id
        train = train.with_columns(
            pl.lit("series_0").alias("unique_id"),
            pl.col("ds").cast(pl.Int64)
        )
        
        schema = DataSchema()
        cfg = SFTConfig()
        artifacts = ArtifactPaths()
        
        train_long = to_long_format(train, schema)
        generate_oos_predictions(train_long, schema, cfg, artifacts)
