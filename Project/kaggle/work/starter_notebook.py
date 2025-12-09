#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import datetime

# Add kaggle_evaluation to path (for local testing)
# In Kaggle environment, this module is pre-installed
try:
    _script_dir = Path(__file__).parent.parent if '__file__' in globals() else None
except:
    _script_dir = None

_kaggle_eval_paths = []
if _script_dir:
    _kaggle_eval_paths.append(_script_dir / "input" / "hull-tactical-market-prediction" / "kaggle_evaluation")
_kaggle_eval_paths.extend([
    Path("/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation"),
    Path("kaggle/input/hull-tactical-market-prediction/kaggle_evaluation"),
])

_kaggle_eval_found = False
for _eval_path in _kaggle_eval_paths:
    if _eval_path.exists():
        sys.path.insert(0, str(_eval_path.parent))
        print(f"  ℹ Added kaggle_evaluation to path: {_eval_path.parent}")
        _kaggle_eval_found = True
        break

if not _kaggle_eval_found and not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print(f"  ℹ kaggle_evaluation not found in local paths (expected in Kaggle environment)")

# ============ Install Required Packages (if not available) ============
def install_from_dataset(package_name: str):
    """
    Try to install package from wheel files in Kaggle input directories.
    """
    import subprocess
    import sys
    import os
    
    wheel_files = []
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            if file.endswith('.whl') and package_name.lower() in file.lower():
                wheel_path = os.path.join(root, file)
                wheel_files.append(wheel_path)
    
    if not wheel_files:
        print(f"  ⚠ No {package_name} wheel files found in /kaggle/input/")
        return False
    
    wheel_files.sort(key=lambda x: (package_name.lower() not in os.path.basename(x).lower(), x))
    
    try:
        wheel_file = wheel_files[0]
        wheel_dir = os.path.dirname(wheel_file)
        print(f"  Installing {package_name} from Dataset: {wheel_file}")
        print(f"  Wheel directory: {wheel_dir}")
        
        all_wheel_files = []
        if os.path.exists(wheel_dir):
            for file in os.listdir(wheel_dir):
                if file.endswith('.whl'):
                    all_wheel_files.append(os.path.join(wheel_dir, file))
        
        print(f"  Found {len(all_wheel_files)} wheel files. Installing compatible packages...")
        all_wheel_files.sort(key=lambda x: "neuralforecast" in os.path.basename(x))
        
        installed_count = 0
        for whl in all_wheel_files:
            pkg_name = os.path.basename(whl).split('-')[0]
            if "numpy" in pkg_name.lower() or "pyarrow" in pkg_name.lower():
                continue
                
            cmd = [sys.executable, "-m", "pip", "install", whl, "--quiet", "--no-index", "--no-deps"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                installed_count += 1
            else:
                if "coreforecast" in pkg_name or "utilsforecast" in pkg_name or "neuralforecast" in pkg_name:
                     print(f"    ✗ Failed to install {os.path.basename(whl)}")
                     print(f"      Error: {result.stderr[:200]}")
                
        print(f"  ✓ Processed all wheels. Installed {installed_count}/{len(all_wheel_files)} packages.")
        
        try:
            import neuralforecast
            print("  ✓ neuralforecast successfully imported")
            return True
        except ImportError as e:
            print(f"  ⚠ neuralforecast import failed: {e}")
            return False
    except Exception as e:
        print(f"  ⚠ Failed to install {package_name} from Dataset: {e}")
        return False

# Try to import neuralforecast
try:
    import neuralforecast
    print("  ✓ neuralforecast is available")
except ImportError:
    print("  ⚠ neuralforecast not found, trying to install from Dataset...")
    if not install_from_dataset("neuralforecast"):
        pass 

import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from neuralforecast import NeuralForecast
import kaggle_evaluation.default_inference_server
import json
import pickle
from collections import deque
from typing import List, Optional
from dataclasses import dataclass, field
import zipfile
import shutil

# ============ CONFIGS ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = 60 

@dataclass
class DataSchema:
    unique_id_col: str = "unique_id"
    timestamp_col: str = "ds"
    target_col: str = "forward_returns"
    # All features used by N-HiTS
    feature_cols: List[str] = field(default_factory=lambda: [
        "E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"
    ])

@dataclass
class ArtifactPaths:
    _local_models = Path("models")
    _kaggle_models_paths = [
        Path("/kaggle/input/models"),
        Path("/kaggle/input/hull-tactical-models/models"),
    ]
    
    def __post_init__(self):
        _base_path = None
        if self._local_models.exists():
            _base_path = str(self._local_models)
        else:
            for kaggle_path in self._kaggle_models_paths:
                if kaggle_path.exists():
                    _base_path = str(kaggle_path)
                    break
        
        if _base_path is None:
            _base_path = str(self._kaggle_models_paths[0])
            
        self.models_dir = _base_path
        self.scaler_path = f"{_base_path}/scaler.pkl"
        self.features_path = f"{_base_path}/features.json"
        self.nhits_save_dir = f"{_base_path}/nhits_checkpoints"
        self.policy_head_weights = f"{_base_path}/policy_head_best.pt"
    
    models_dir: str = ""
    scaler_path: str = ""
    features_path: str = ""
    nhits_save_dir: str = ""
    policy_head_weights: str = ""

# ============ COMPONENTS ============

@dataclass
class FeatureStore:
    schema: DataSchema
    scaler: Optional[StandardScaler] = None
    feature_order: Optional[List[str]] = None

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.scaler is None: raise RuntimeError("FeatureStore not loaded")
        df_pd = df.to_pandas()
        if not all(c in df_pd.columns for c in self.feature_order):
             for c in self.feature_order:
                 if c not in df_pd.columns:
                     df_pd[c] = 0.0
        
        df_pd[self.feature_order] = self.scaler.transform(df_pd[self.feature_order].astype(float))
        return pl.from_pandas(df_pd)

    def load(self, paths: ArtifactPaths) -> None:
        with open(paths.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(paths.features_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_order = meta["feature_order"]

class PolicyHead(nn.Module):
    """V2 Policy Head: Bernoulli (Direction) + Beta (Magnitude)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.dir_head = nn.Linear(hidden_dim // 2, 1)
        self.mag_head = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        p_long = torch.sigmoid(self.dir_head(h)).squeeze(-1)
        mag_raw = self.mag_head(h)
        alpha = F.softplus(mag_raw[:, 0]) + 1.0 + 1e-6
        beta = F.softplus(mag_raw[:, 1]) + 1.0 + 1e-6
        return p_long, alpha, beta

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        p_long, alpha, beta = self.forward(x)
        if deterministic:
            dir_taken = (p_long > 0.5).float()
            mag = alpha / (alpha + beta)
        else:
            dir_taken = torch.bernoulli(p_long)
            dist = torch.distributions.Beta(alpha, beta)
            mag = dist.rsample()
        position = dir_taken * (2.0 * mag)
        return position.clamp(0.0, 2.0)

# ============ RUNTIME STATE ============

class Runtime:
    loaded = False
    feature_store: Optional[FeatureStore] = None
    feature_cols: Optional[List[str]] = None
    nhits_nf: Optional[NeuralForecast] = None
    policy_head: Optional[PolicyHead] = None
    history: Optional[deque] = None
    y_hat_history: Optional[deque] = None
    return_history: Optional[deque] = None
    last_prediction_debug: Optional[dict] = None  # For debugging

runtime = Runtime()
schema = DataSchema()
artifacts = ArtifactPaths()

def load_artifacts():
    if runtime.loaded: return
    print("Loading Artifacts...")
    
    # 1. Feature Store
    runtime.feature_store = FeatureStore(schema=schema)
    runtime.feature_store.load(artifacts)
    runtime.feature_cols = runtime.feature_store.feature_order
    
    # 2. N-HiTS
    nhits_zip = Path(artifacts.models_dir) / "nhits_checkpoints.zip"
    nhits_dir = Path(artifacts.nhits_save_dir)
    
    if not nhits_dir.exists() and nhits_zip.exists():
        print(f"  Extracting {nhits_zip}...")
        with zipfile.ZipFile(nhits_zip, 'r') as z:
            z.extractall(Path(artifacts.models_dir))
            
    nested = Path(artifacts.models_dir) / "nhits_checkpoints" / "nhits_checkpoints"
    if nested.exists(): nhits_dir = nested
        
    runtime.nhits_nf = NeuralForecast.load(path=str(nhits_dir))
    print("  ✓ N-HiTS Loaded")
    
    # 3. Policy Head (Input Dim = 7 stats + 2 Raw Feats = 9)
    runtime.policy_head = PolicyHead(input_dim=9, hidden_dim=128)
    ph_path = Path(artifacts.policy_head_weights)
    if ph_path.exists():
        state_dict = torch.load(ph_path, map_location=DEVICE)
        runtime.policy_head.load_state_dict(state_dict)
        runtime.policy_head.to(DEVICE).eval()
        print("  ✓ PolicyHead V2 Loaded")
    else:
        print(f"  ⚠ PolicyHead not found at {ph_path}")
        runtime.policy_head = None
        
    # 4. Init Buffers
    runtime.history = deque(maxlen=WINDOW)
    runtime.y_hat_history = deque(maxlen=20)
    runtime.return_history = deque(maxlen=20)
    
    runtime.loaded = True

# ============ INFERENCE LOGIC ============

def _cold_start_strategy(test_row: pl.DataFrame) -> float:
    """Momentum based cold start"""
    if "lagged_forward_returns" in test_row.columns:
        val = test_row["lagged_forward_returns"][0]
        if val is not None and val == val: 
            import math
            return 0.8 + 0.4 * math.tanh(float(val) * 50.0)
    return 0.8

def prepare_input(test_row: pl.DataFrame) -> Optional[pl.DataFrame]:
    row = test_row.clone()
    cols = runtime.feature_cols
    
    for c in cols:
        if c in row.columns:
            row = row.with_columns(pl.col(c).fill_null(0.0))
        else:
            row = row.with_columns(pl.lit(0.0).alias(c))
            
    runtime.history.append(row.to_dicts()[0])
    
    if len(runtime.history) < WINDOW:
        return None
        
    hist_df = pl.DataFrame(list(runtime.history))
    feat_df = hist_df.select(cols)
    feat_scaled = runtime.feature_store.transform(feat_df)
    
    nf_df = pl.DataFrame({
        "unique_id": ["series_0"] * WINDOW,
        "ds": list(range(WINDOW)),
        "y": [0.0] * WINDOW 
    })
    
    for i, c in enumerate(cols):
        nf_df = nf_df.with_columns(pl.Series(c, feat_scaled[:, i].to_numpy()))
        
    return nf_df

@torch.inference_mode()
def predict_position(window_df: pl.DataFrame, test_row: pl.DataFrame) -> float:
    # 1. N-HiTS Prediction
    window_pd = window_df.to_pandas()
    test_feat = test_row.select(runtime.feature_cols)
    for c in runtime.feature_cols:
        if c not in test_feat.columns:
            test_feat = test_feat.with_columns(pl.lit(0.0).alias(c))
    test_feat = test_feat.fill_null(0.0)
    test_scaled = runtime.feature_store.transform(test_feat)
    
    futr_df = pd.DataFrame({
        "unique_id": ["series_0"],
        "ds": [WINDOW]
    })
    for i, c in enumerate(runtime.feature_cols):
        futr_df[c] = test_scaled[:, i].to_numpy()[0]
        
    fcst = runtime.nhits_nf.predict(df=window_pd, futr_df=futr_df)
    nhits_y_hat = float(fcst['NHITS'].values[0])
    
    # 2. Update Stats Buffer
    runtime.y_hat_history.append(nhits_y_hat)
    if "lagged_forward_returns" in test_row.columns:
        val = test_row["lagged_forward_returns"][0]
        if val is not None: runtime.return_history.append(float(val))
             
    # 3. Calc Stats
    if len(runtime.return_history) >= 2:
        rh = np.array(runtime.return_history)
        roll_vol = float(np.std(rh))
        roll_trend = float(np.mean(rh))
    else:
        roll_vol, roll_trend = 0.0, 0.0
        
    if len(runtime.y_hat_history) >= 2:
        yh = np.array(runtime.y_hat_history)
        yhat_ema = float(np.mean(yh))
        yhat_std = float(np.std(yh)) + 1e-6
        y_hatz = (nhits_y_hat - yhat_ema) / yhat_std
        conf_abs = abs(y_hatz)
    else:
        y_hatz, yhat_ema, yhat_std, conf_abs = 0.0, nhits_y_hat, 1.0, 0.0
        
    # 4. Policy Head Inference
    if runtime.policy_head is None:
        return 2.0 * (1.0 / (1.0 + np.exp(-nhits_y_hat)))
    
    # Extract Raw P8, S2
    p8_val = float(test_row["P8"][0]) if "P8" in test_row.columns else 0.0
    s2_val = float(test_row["S2"][0]) if "S2" in test_row.columns else 0.0
    if p8_val is None: p8_val = 0.0
    if s2_val is None: s2_val = 0.0
    
    # [nhits_y_hat, vol, trend, y_hatz, yhat_ema, yhat_std, conf_abs, P8, S2]
    state = np.array([
        nhits_y_hat, roll_vol, roll_trend,
        y_hatz, yhat_ema, yhat_std, conf_abs,
        p8_val, s2_val
    ], dtype=np.float32)
    
    state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
    action = runtime.policy_head.get_action(state_t, deterministic=True)
    
    # Store debug info
    runtime.last_prediction_debug = {
        "nhits_y_hat": nhits_y_hat,
        "roll_vol": roll_vol,
        "roll_trend": roll_trend,
        "y_hatz": y_hatz,
        "yhat_ema": yhat_ema,
        "yhat_std": yhat_std,
        "conf_abs": conf_abs,
        "p8_raw": p8_val,
        "s2_raw": s2_val,
        "state_vector": state.tolist(),
        "final_action": float(action.item())
    }
    
    return float(action.item())

def predict(test: pl.DataFrame) -> float:
    if not runtime.loaded: load_artifacts()
    
    if len(test) > 1:
        if "date_id" in test.columns:
            test = test.sort("date_id", descending=True).head(1)
        else:
            test = test.head(1)
            
    window_df = prepare_input(test)
    
    if window_df is None:
        return _cold_start_strategy(test)
        
    try:
        return predict_position(window_df, test)
    except Exception as e:
        # print(f"Inference Error: {e}")
        return _cold_start_strategy(test)

# ============ SERVER ============
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    print("Local Testing Mode")
    # Try Kaggle absolute path first (for online notebook)
    kaggle_path = "/kaggle/input/hull-tactical-market-prediction/"
    # Try local relative path (for local dev)
    local_path = "kaggle/input/hull-tactical-market-prediction/"
    
    data_path = None
    if os.path.exists(kaggle_path):
        data_path = kaggle_path
    elif os.path.exists(local_path):
        data_path = local_path
        
    if data_path:
        print(f"  Found data at: {data_path}")
        inference_server.run_local_gateway(
            (data_path,)
        )
    else:
        print("  ⚠ Test data not found. Cannot run local gateway.")