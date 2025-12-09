import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.policy_head import PolicyHead
from src.configs import ArtifactPaths, DataSchema

# -----------------------------------------------------------------------------
# 1. Reward Function (Comprehensive Version V2)
# -----------------------------------------------------------------------------
def compute_reward(
    position: torch.Tensor,           # Current position [B] [0, 2]
    returns: torch.Tensor,            # Actual returns [B] (forward_returns[T])
    volatility: torch.Tensor,         # Volatility [B]
    y_hat_conf: torch.Tensor,         # Confidence signal [B] (abs(z-score))
    scale: float = 2000.0,            # Balanced: strong signal but not too large
    alpha_mag: float = 20.0,          # Moderate emphasis on magnitude
    lambda_vol: float = 0.0,          # Disable risk penalty
    lambda_turn: float = 0.001        # Small turnover penalty for stability
) -> torch.Tensor:
    """
    Enhanced Reward Function V2:
    Focuses on Direction * Magnitude, weighted by Confidence.
    """
    
    # 1. Direction Logic
    # position > 0 is considered "Long-ish" (since range is [0,2], neutral is 1)
    # But wait, original code treated [0, 2] as Long/Short? 
    # Let's align: 
    # Usually 2=Long, 0=Short, 1=Neutral. 
    # Let's standardize: Action > 1.0 => Long, Action < 1.0 => Short.
    # Center action around 0 for direction calculation: (action - 1.0)
    action_centered = position - 1.0 # range [-1, 1]
    
    dir_taken = torch.sign(action_centered) # -1, 0, 1
    sign_y = torch.sign(returns)           # -1, 0, 1
    
    # +1 if direction matches, -1 if opposite, 0 if neutral
    reward_dir = dir_taken * sign_y 
    
    # Asymmetric penalty: punish wrong direction more heavily
    # This encourages the model to be more careful
    reward_dir = torch.where(
        reward_dir > 0,
        reward_dir * 1.0,      # Correct: keep as is
        reward_dir * 1.5       # Wrong: 1.5x penalty (not too harsh)
    )

    # 2. Magnitude Logic
    # Reward is proportional to the absolute return size
    # "If I'm right on a big move, I get huge reward. If wrong on big move, huge penalty."
    reward_mag = torch.abs(returns)
    
    # Combined: Direction * (1 + alpha * Magnitude)
    # Base + Bonus for catching big moves
    combined = reward_dir * (1.0 + alpha_mag * reward_mag)

    # 3. Confidence Scaling
    # Use sigmoid to map confidence (z-score) to [0.5, 1.5] range approx
    # High confidence => Higher reward/penalty
    # conf = torch.sigmoid(5.0 * y_hat_conf) # [0, 1]
    # Let's keep it simple: just multiply? Or V2 logic:
    conf = torch.sigmoid(5.0 * y_hat_conf)
    combined = combined * (0.5 + conf) 

    # 4. Scale Up
    combined = combined * scale

    # 5. Penalties
    # Risk: Penalize holding large positions in high vol
    # Use centered action for penalty to penalize deviation from neutral
    risk_pen = lambda_vol * (action_centered * volatility) ** 2
    
    # Turnover: Penalize absolute position size (proxy for turnover/risk exposure)
    turn_pen = lambda_turn * torch.abs(action_centered)

    reward = combined - risk_pen - turn_pen
    return reward

# -----------------------------------------------------------------------------
# 2. Dataset Definition (Enhanced V2)
# -----------------------------------------------------------------------------
class GRPODataset(Dataset):
    def __init__(self, y_hat_path: str, raw_csv_path: str, feature_cols: list, window_size: int = 20):
        # Load OOS signals (nhits_y_hat is the one we want)
        self.signals = pl.read_parquet(y_hat_path)
        
        # [ALIGNMENT] Shift signals forward by 1 so y_hat[T] predicts Returns[T]
        self.signals = self.signals.with_columns(
            (pl.col("ds") + 1).alias("ds")
        )
        
        # Load Raw Data
        self.raw = pl.read_csv(raw_csv_path)
        if "date_id" in self.raw.columns:
            self.raw = self.raw.with_columns(pl.col("date_id").cast(pl.Int64).alias("ds"))
        if "forward_returns" in self.raw.columns:
            self.raw = self.raw.rename({"forward_returns": "y"})
            
        # [Cleaning]
        # Filter: Require ALL selected features to be present (null_count == 0)
        # Since we only selected 2 high-quality features, we can be strict.
        original_len = len(self.raw)
        null_counts = self.raw.select(
            pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
        )
        self.raw = self.raw.with_columns(null_counts).filter(
            (pl.col("null_count") == 0) & (pl.col("y").is_not_null())
        ).drop("null_count")
        
        # Join
        self.df = self.raw.join(self.signals, on="ds", how="inner").sort("ds")
        
        # [Cleaning post-join]
        self.df = self.df.filter(pl.col("nhits_y_hat").is_not_null())
        
        # [Feature Engineering V2]
        # 1. Rolling Stats for y (Market State)
        self.df = self.df.with_columns([
            pl.col("y").rolling_std(window_size).alias("rolling_vol").fill_null(0.0),
            pl.col("y").rolling_mean(window_size).alias("rolling_trend").fill_null(0.0),
        ])
        
        # 2. Advanced Stats for y_hat (Signal State)
        # Z-score: (y_hat - mean) / std
        # EMA: Exponential Moving Average (approximated by rolling mean here for simplicity/speed)
        self.df = self.df.with_columns([
            pl.col("nhits_y_hat").rolling_mean(window_size).alias("yhat_ema").fill_null(0.0),
            pl.col("nhits_y_hat").rolling_std(window_size).alias("yhat_std").fill_null(1e-6),
        ])
        
        self.df = self.df.with_columns([
            ((pl.col("nhits_y_hat") - pl.col("yhat_ema")) / pl.col("yhat_std")).alias("y_hatz"),
        ])
        
        self.df = self.df.with_columns([
            pl.col("y_hatz").abs().alias("conf_abs").fill_null(0.0) # Absolute confidence
        ])

        # State Construction:
        # 1. Signal: nhits_y_hat
        # 2. Market: vol, trend
        # 3. Signal Stats: y_hatz, yhat_ema, yhat_std, conf_abs
        # 4. Raw Features: E2, M13... (Keep them!)
        self.state_cols = [
            "nhits_y_hat", "rolling_vol", "rolling_trend", 
            "y_hatz", "yhat_ema", "yhat_std", "conf_abs"
        ] + feature_cols
        
        # Fill NaNs
        self.df = self.df.with_columns([
            pl.col(c).fill_null(0.0) for c in self.state_cols
        ])
        
        # Convert to numpy
        self.state_data = self.df.select(self.state_cols).to_numpy().astype(np.float32)
        
        # Target Data: [y_true, vol, conf_abs]
        self.target_data = self.df.select(["y", "rolling_vol", "conf_abs"]).to_numpy().astype(np.float32)
        
    def __len__(self):
        return len(self.state_data)
    
    def __getitem__(self, idx):
        return self.state_data[idx], self.target_data[idx]

# -----------------------------------------------------------------------------
# 3. Training Script
# -----------------------------------------------------------------------------
def train_grpo():
    artifacts = ArtifactPaths()
    schema = DataSchema()
    
    signals_path = Path(artifacts.models_dir) / "sft_y_hat_oos.parquet"
    # Use train_feature_selected.csv
    raw_data_path = Path("kaggle/input/hull-tactical-market-prediction/train_feature_selected.csv")
    if not raw_data_path.exists():
         raw_data_path = Path("/kaggle/input/hull-tactical-market-prediction/train_feature_selected.csv")
    
    if not signals_path.exists():
        print("Signals file not found. Run gen_sft_signals.py first.")
        return

    # Configuration
    EPOCHS = 150  # Increased for better convergence
    BATCH_SIZE = 128
    LR = 5e-4 # Slightly lower LR for stability with new Reward
    
    # Prepare Data
    # Feature Selection: Use only "P8" and "S2" (Best Rank IC + Data Quality)
    selected_features = ["P8", "S2"]
    dataset = GRPODataset(
        str(signals_path), 
        str(raw_data_path),
        feature_cols=selected_features,
        window_size=20
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    # Input dim = 7 basic stats + 2 raw features = 9
    input_dim = 7 + len(selected_features)
    policy = PolicyHead(input_dim=input_dim, hidden_dim=128)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    # Track Best Model
    best_reward = float('-inf')
    best_model_path = Path(artifacts.models_dir) / "policy_head_best.pt"
    
    # Store history for plotting
    history = {"reward": [], "loss": []}
    
    print(f"Starting GRPO Training (V2 Logic)...")
    print(f"State Dim: {input_dim}")
    
    for epoch in range(EPOCHS):
        total_reward = 0
        total_loss = 0
        
        for states, targets in dataloader:
            # ... (training loop remains same) ...
            # 1. Sample Actions
            actions = policy.get_action(states, deterministic=False)
            
            # 2. Compute Reward
            y_true = targets[:, 0]
            vol = targets[:, 1]
            conf_abs = targets[:, 2]
            
            rewards = compute_reward(
                position=actions,
                returns=y_true,
                volatility=vol,
                y_hat_conf=conf_abs,
                scale=2000.0,       # Balanced signal strength
                alpha_mag=20.0,     # Moderate magnitude emphasis
                lambda_vol=0.0,
                lambda_turn=0.001   # Small turnover penalty
            )
            
            # 3. Update
            baseline = rewards.mean()
            adv = rewards - baseline
            
            # [Stabilization] Advantage Normalization
            # Normalize advantage to mean 0, std 1 to stabilize gradients
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
        avg_rew = total_reward / len(dataloader)
        history["reward"].append(avg_rew)
        history["loss"].append(total_loss)
        
        # Save Best Model (Based on Reward)
        is_best = ""
        if avg_rew > best_reward:
            best_reward = avg_rew
            torch.save(policy.state_dict(), best_model_path)
            is_best = "★"
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Reward: {avg_rew:.4f} | Loss: {total_loss:.4f} {is_best}")
        
    # Save Final
    save_path = Path(artifacts.models_dir) / "policy_head.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Training Complete. Model saved to {save_path}")
    
    # Save Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history["reward"], label="Reward")
        plt.title("Training Reward")
        plt.xlabel("Epoch")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history["loss"], label="Loss", color="orange")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(artifacts.models_dir) / "grpo_training_curve.png")
        print("Training curve saved to models/grpo_training_curve.png")
    except ImportError:
        print("Matplotlib not found, skipping plot.")

if __name__ == "__main__":
    train_grpo()