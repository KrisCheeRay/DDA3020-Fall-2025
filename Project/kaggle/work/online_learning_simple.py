#!/usr/bin/env python
"""
Online Learning Strategy: Simple MLP with incremental updates
- Uses 8 raw features + engineered features
- Trains on the fly as data arrives
- No need for pre-trained models
"""

import os
import sys
from pathlib import Path
from collections import deque

# Add kaggle_evaluation to path
_kaggle_eval_paths = [
    Path("/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation"),
    Path("kaggle/input/hull-tactical-market-prediction/kaggle_evaluation"),
]

for _eval_path in _kaggle_eval_paths:
    if _eval_path.exists():
        sys.path.insert(0, str(_eval_path.parent))
        print(f"✓ Added kaggle_evaluation to path: {_eval_path.parent}")
        break

import kaggle_evaluation.default_inference_server
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Configuration
# =============================================================================
FEATURE_COLS = ["E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"]
WINDOW_SIZE = 30  # History window for online learning
BATCH_UPDATE = 5   # Update model every N samples
HIDDEN_DIM = 64
LR = 1e-3


# =============================================================================
# Simple MLP Model
# =============================================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Predict forward return
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# Online Learning Runtime
# =============================================================================
class OnlineRuntime:
    def __init__(self):
        # Data buffers
        self.feature_history = deque(maxlen=WINDOW_SIZE)
        self.return_history = deque(maxlen=WINDOW_SIZE)
        
        # Model (will be initialized on first valid sample)
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Feature statistics (for standardization)
        self.feature_mean = None
        self.feature_std = None
        
        # Training buffer
        self.train_buffer_X = []
        self.train_buffer_y = []
        
        # Counter
        self.sample_count = 0
        
    def clean_row(self, row: dict) -> dict:
        """Data cleaning: Fill nulls with 0, check validity"""
        null_count = sum(1 for col in FEATURE_COLS if row.get(col) is None or np.isnan(row.get(col, 0.0)))
        
        if null_count > 4:
            return None  # Too many nulls, skip
        
        # Fill remaining nulls with 0
        cleaned = {}
        for col in FEATURE_COLS:
            val = row.get(col, 0.0)
            cleaned[col] = 0.0 if (val is None or np.isnan(val)) else float(val)
            
        return cleaned
    
    def engineer_features(self, raw_features: dict) -> np.ndarray:
        """Feature engineering"""
        base_features = [raw_features[col] for col in FEATURE_COLS]
        
        # Engineered features
        features = base_features.copy()
        
        # Rolling statistics (if we have history)
        if len(self.feature_history) > 0:
            history_arr = np.array([[h[col] for col in FEATURE_COLS] for h in self.feature_history])
            
            # Mean of last 5
            if len(self.feature_history) >= 5:
                recent_mean = np.mean(history_arr[-5:], axis=0)
                features.extend(recent_mean.tolist())
            else:
                features.extend([0.0] * len(FEATURE_COLS))
            
            # Std of last 10
            if len(self.feature_history) >= 10:
                recent_std = np.std(history_arr[-10:], axis=0)
                features.extend(recent_std.tolist())
            else:
                features.extend([0.0] * len(FEATURE_COLS))
        else:
            features.extend([0.0] * len(FEATURE_COLS) * 2)
        
        return np.array(features, dtype=np.float32)
    
    def standardize(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using running statistics"""
        if self.feature_mean is None:
            # First sample: initialize
            self.feature_mean = features.copy()
            self.feature_std = np.ones_like(features)
            return np.zeros_like(features)
        else:
            # Update running statistics (exponential moving average)
            alpha = 0.05
            self.feature_mean = (1 - alpha) * self.feature_mean + alpha * features
            diff = (features - self.feature_mean) ** 2
            self.feature_std = np.sqrt((1 - alpha) * self.feature_std ** 2 + alpha * diff)
            
            # Standardize
            return (features - self.feature_mean) / (self.feature_std + 1e-6)
    
    def predict_return(self, features: np.ndarray) -> float:
        """Predict forward return using model"""
        if self.model is None:
            # Initialize model on first valid sample
            input_dim = len(features)
            self.model = SimpleMLP(input_dim, HIDDEN_DIM)
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
            print(f"✓ Model initialized with input_dim={input_dim}")
            
            # Return neutral prediction initially
            return 0.0
        
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            pred = self.model(x).item()
        
        return pred
    
    def update_model(self, features: np.ndarray, true_return: float):
        """Update model with new observation"""
        if self.model is None:
            return  # Not initialized yet
        
        # Add to buffer
        self.train_buffer_X.append(features)
        self.train_buffer_y.append(true_return)
        
        # Update every BATCH_UPDATE samples
        if len(self.train_buffer_X) >= BATCH_UPDATE:
            self.model.train()
            
            X = torch.from_numpy(np.array(self.train_buffer_X))
            y = torch.tensor(self.train_buffer_y, dtype=torch.float32)
            
            # Train for 1 step
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            
            # Clear buffer
            self.train_buffer_X = []
            self.train_buffer_y = []
    
    def process(self, test_row: dict, lagged_return: float) -> float:
        """Main processing: predict current, then update with previous truth"""
        self.sample_count += 1
        
        # 1. Clean current row
        cleaned = self.clean_row(test_row)
        if cleaned is None:
            return 1.0  # Default neutral if too many nulls
        
        # 2. Update model with PREVIOUS observation (if available)
        if len(self.feature_history) > 0 and len(self.return_history) > 0:
            prev_features = self.feature_history[-1]
            prev_return = self.return_history[-1]
            
            # Engineer and standardize previous features
            prev_feat_eng = self.engineer_features(prev_features)
            prev_feat_std = self.standardize(prev_feat_eng)
            
            # Update model
            self.update_model(prev_feat_std, prev_return)
        
        # 3. Engineer features for CURRENT row
        features_eng = self.engineer_features(cleaned)
        features_std = self.standardize(features_eng)
        
        # 4. Predict return for current row
        pred_return = self.predict_return(features_std)
        
        # 5. Convert return to position [0, 2]
        # Use tanh to map to [-1, 1], then shift to [0, 2]
        position = 1.0 + np.tanh(pred_return * 100.0)  # Scale by 100 for sensitivity
        position = np.clip(position, 0.0, 2.0)
        
        # 6. Store current features (will be updated with true return next time)
        self.feature_history.append(cleaned)
        
        # 7. Store lagged return as proxy for true return of PREVIOUS period
        # (In real online setting, lagged_forward_returns gives us the truth)
        if lagged_return is not None and not np.isnan(lagged_return):
            self.return_history.append(float(lagged_return))
        
        return float(position)


# =============================================================================
# Global runtime
# =============================================================================
runtime = OnlineRuntime()


def predict(test: pl.DataFrame) -> float:
    """Main prediction function called by Kaggle"""
    # Extract single row
    if len(test) > 1:
        if "date_id" in test.columns:
            test = test.sort("date_id", descending=True).head(1)
        else:
            test = test.head(1)
    
    row = test.to_dicts()[0]
    
    # Get lagged return (this is the "truth" for previous period)
    lagged_return = row.get("lagged_forward_returns", None)
    
    # Process
    position = runtime.process(row, lagged_return)
    
    return position


# =============================================================================
# Server
# =============================================================================
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("Running in Kaggle competition mode (online learning)...")
    inference_server.serve()
else:
    print("="*60)
    print("ONLINE LEARNING TEST")
    print(f"Features: {FEATURE_COLS}")
    print(f"Window: {WINDOW_SIZE} | Batch: {BATCH_UPDATE}")
    print(f"Model: MLP({HIDDEN_DIM}) | LR: {LR}")
    print("="*60)
    
    # Local testing
    kaggle_path = "/kaggle/input/hull-tactical-market-prediction/"
    local_path = "kaggle/input/hull-tactical-market-prediction/"
    
    if os.path.exists(kaggle_path):
        data_path = kaggle_path
    elif os.path.exists(local_path):
        data_path = local_path
    else:
        print("✗ Data path not found")
        sys.exit(1)
        
    print(f"Data path: {data_path}\n")
    inference_server.run_local_gateway((data_path,))
    
    print("\n" + "="*60)
    print(f"Online learning complete! Processed {runtime.sample_count} samples.")
    print("="*60)

