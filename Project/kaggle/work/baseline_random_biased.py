#!/usr/bin/env python
"""
Baseline Strategy: Random output in [0.7, 1.3] with bias towards >1.0
This tests if random trading with slight bullish bias can get a good score.
"""

import os
import sys
from pathlib import Path
import numpy as np

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

# Set random seed for reproducibility
np.random.seed(42)


def predict(test: pl.DataFrame) -> float:
    """
    Random strategy with bullish bias:
    - Range: [0.7, 1.3]
    - Probability: 60% for >1.0, 40% for <1.0
    
    This simulates a "dumb trader" who leans bullish but randomly.
    """
    # Use Beta distribution to create bias
    # alpha=2, beta=1.5 creates a distribution skewed towards higher values
    # Then scale to [0.7, 1.3] range
    
    # Generate a value in [0, 1] with bias towards higher values
    raw_sample = np.random.beta(a=2.0, b=1.5)
    
    # Scale to [0.7, 1.3]
    position = 0.7 + raw_sample * 0.6
    
    return float(position)


# Create inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("Running in Kaggle competition mode...")
    inference_server.serve()
else:
    print("="*60)
    print("BASELINE TEST: Random [0.7-1.3] with bullish bias")
    print("Beta(2.0, 1.5) distribution → ~60% above 1.0")
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
        
    print(f"Data path: {data_path}")
    
    # Show distribution
    print("\nSample distribution (100 samples):")
    samples = [0.7 + np.random.beta(2.0, 1.5) * 0.6 for _ in range(100)]
    print(f"  Mean: {np.mean(samples):.3f}")
    print(f"  Std:  {np.std(samples):.3f}")
    print(f"  Min:  {np.min(samples):.3f}")
    print(f"  Max:  {np.max(samples):.3f}")
    print(f"  Above 1.0: {sum(s > 1.0 for s in samples)}%")
    print()
    
    # Reset seed for actual inference
    np.random.seed(42)
    
    inference_server.run_local_gateway((data_path,))
    
    print("\n" + "="*60)
    print("Random baseline test complete!")
    print("Compare with:")
    print("  - Constant 1.0 baseline")
    print("  - Your sophisticated model")
    print("="*60)

