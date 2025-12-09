# End-to-End Architecture Deep Dive

This document provides a code-level, step-by-step analysis of the **SFT (Supervised Fine-Tuning) + GRPO (Policy Optimization)** dual-stage architecture used in this project.

---

## 0. The Big Picture (Global Data Flow)

The entire pipeline consists of four phases:

1.  **Phase 0: Data Prep**
    *   **Input**: Raw CSV (Wide Format).
    *   **Action**: Cleaning, feature engineering, standardization (`FeatureStore`).
    *   **Output**: Long Format data conforming to `DataSchema`.
2.  **Phase 1: Signal Generation**
    *   **Input**: Processed training data.
    *   **Action**: `NeuralForecast` Cross Validation (Rolling CV) + Ensemble.
    *   **Output**: `sft_y_hat_oos.parquet` ("Rotten Apple" data).
3.  **Phase 2: Policy Training**
    *   **Input**: `sft_y_hat_oos.parquet`.
    *   **Action**: Train Policy Head (MLP) using GRPO.
    *   **Output**: `policy_head_best.pt`.
4.  **Phase 3: Online Inference**
    *   **Input**: Real-time market data.
    *   **Action**: Feature transformation -> SFT Inference -> Policy Decision.

---

## 0.5 Data Interface & Feature Store (`src/feature_store.py`)

Before entering complex model training, we must unify the data "language".

### Interface Contract (`src/configs.py`)
All models strictly adhere to the following column names:
*   `unique_id`: Series identifier (Kaggle competitions usually have only one series, e.g., 'series_0').
*   `ds`: Timestamp (Date/Time Step).
*   `y`: Prediction target (Forward Returns).
*   `features`: All other columns.

### Role of FeatureStore
We cannot allow data distribution inconsistency between training and inference. `FeatureStore` guarantees this.

```python
# Training (Fit & Transform)
fs = FeatureStore(schema)
train_scaled = fs.fit_transform(train_df) 
# -> Calculates mean/std, saves to scaler.pkl, records column order to features.json

# Inference/Validation (Transform Only)
fs.load(artifacts)
test_scaled = fs.transform(test_df)
# -> Uses previously saved mean/std for transformation and enforces column order alignment
```

---

## 1. Phase 1: Rotten Apple Factory (`kaggle/work/gen_sft_signals.py`)

The core task of this script is to **"honestly simulate errors"**. We use `NeuralForecast`'s `cross_validation` function to implement efficient rolling prediction.

### Core Code Analysis

```python
def generate_oos_signals(df, ...):
    # 1. Global Feature Scaling
    # Note: PatchTST is insensitive to scaling (RevIN), but for consistency, we do a transform first.
    fs = FeatureStore(schema)
    df_scaled = fs.fit_transform(df)
    
    # 2. Cross Validation (Rolling Cross Validation / Walk-Forward Validation)
    # 
    # [Correct Understanding: The Right Way for Time Series Cross Validation]
    # The correct behavior of cross_validation (regardless of refit=True or False):
    #   - Window 1: Train[0:60] -> Train Model -> Predict Test[61]
    #   - Window 2: Train[0:61] -> Train Model -> Predict Test[62]
    #   - Window 3: Train[0:62] -> Train Model -> Predict Test[63]
    #   - ...
    #   Critical: The test set for each window is "future data", no leakage!
    #
    # Difference between refit=False vs refit=True:
    #   - refit=True: Each window trains from random initialization (slow, but more independent)
    #   - refit=False: Each window might reuse weights from the previous window (fast, but potentially correlated)
    #   But in any case, each window trains on its own training set, never using "future data"!
    #
    # Why "Rotten Apples"?
    #   - Even if retrained every window, data distribution might drift over time
    #   - Models trained on "old data" predicting "new data" will accumulate errors
    #   - This simulates the real scenario: models aren't retrained frequently, errors accumulate
    cv_df = nf.cross_validation(
        df=df_scaled.to_pandas(),
        n_windows=total_len - input_size, # Covers almost all history
        step_size=1,                      # Rolling daily, each window predicts 1 step
        refit=False                       # Reusing weights for speed, maintains train/test separation
    )
    
    # 3. Dynamic Ensemble
    # Simulate inference behavior: get predictions from PatchTST and NHITS, weight by volatility.
    # This step is crucial to ensure the Policy Head sees the same input distribution as online.
    vol = compute_recent_vol(cv_df) # Calculate environmental volatility
    final_y_hat = combine_predictions(
        pred_patch=cv_df["PatchTST"],
        pred_nhits=cv_df.get("NHITS"),
        vol=vol,
        config=ensemble_cfg
    )
    
    # 4. Save Rotten Apples
    # Only these y_hats with real errors are qualified feed for GRPO training.
    result_df.write_parquet("models/sft_y_hat_oos.parquet")
```

**Key Point: Ensemble Consistency**
*   **Principle**: Train as you serve.
*   If you use `0.6*A + 0.4*B` in `predict_runtime.py`, you must generate training data using the same logic here.
*   Never feed pure PatchTST results to GRPO and then give it hybrid model results online.

---

## 2. Phase 2: The Decision Brain (`src/policy_head.py`)

This file defines our decision agent. It is a simple Multi-Layer Perceptron (MLP), but the output layer is specific.

### Core Code Analysis

```python
class PolicyHead(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        # Input layer: Why 3?
        # Because we feed it: [SFT Ensemble Prediction, Historical Volatility, Historical Trend]
        # These three numbers form the entire basis for its decision.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # ... hidden layers omitted ...
            nn.Linear(hidden_dim // 2, 2)  # Output layer: outputs 2 numbers
        )
```

#### Key Point: How to output continuous action [0, 2]?

We cannot directly output a scalar because we need a **probability distribution** to support Reinforcement Learning exploration. We choose the **Beta Distribution**.

```python
    def forward(self, x):
        out = self.net(x)
        # Softplus ensures parameters > 0
        # +1.0 makes the Beta distribution tend towards unimodal, training is more stable
        alpha = F.softplus(out[:, 0]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 1]) + 1.0 + 1e-6
        return alpha, beta
```

#### Action Sampling (Training vs Inference)

```python
    def get_action(self, x, deterministic=False):
        alpha, beta = self.forward(x)
        
        if deterministic:
            # Inference Mode (Online): Use Mean
            # Beta Mean = alpha / (alpha + beta) -> [0, 1]
            action = 2.0 * (alpha / (alpha + beta))
        else:
            # Training Mode (GRPO): Sample
            # Even if mean is 1.0, might sample 1.2 or 0.8 for exploration
            dist = torch.distributions.Beta(alpha, beta)
            action = 2.0 * dist.sample()
            
        return action
```

---

## 3. Phase 3: Training Dojo (`kaggle/work/train_grpo.py`)

This is where the magic happens: how the model learns to make money without "correct answers".

### Dataset Construction (`GRPODataset`)

```python
    def __getitem__(self, idx):
        # State: Information the model sees right now (y_hat, vol)
        state = self.data[idx, :state_dim]
        # Target: Information the model cannot see now, but used later for grading (y_true)
        target_info = self.data[idx, target_idx]
        return state, target_info
```

### Training Loop (The Loop)

```python
        for states, targets in dataloader:
            # 1. Decision (Actor)
            # Policy Head looks at state and uncertainly gives actions
            actions = policy.get_action(states, deterministic=False)
            
            # 2. Grading (Critic/Environment)
            # Market calculates PnL based on actions and y_true
            rewards = compute_reward(actions, y_true, vol)
            
            # 3. Calculate Advantage
            # How much better than average?
            adv = rewards - rewards.mean()
            
            # 4. Policy Gradient
            # Loss = - log_prob(action) * advantage
            # Increase prob if profitable, decrease if loss
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            # 5. Correct the Brain
            optimizer.step()
```

---

## 4. Summary: Sim-to-Real Loop

1.  **Phase 0**: Ensures consistency of training and inference features.
2.  **Phase 1**: Creates signals with realistic errors and ensemble characteristics, preventing "overfitting to perfect predictions".
3.  **Phase 2/3**: Directly optimizes PnL, teaching the model how to control positions when predictions are inaccurate (typically learning to leverage when volatility is low and reduce positions when predictions are fuzzy).

This architecture has strong robustness in Kaggle financial competitions.
