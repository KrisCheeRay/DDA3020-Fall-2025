import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.policy_head import PolicyHead
from src.configs import ArtifactPaths

# -----------------------------------------------------------------------------
# 1. 奖励函数 (由量化同学提供，此处为示例)
# -----------------------------------------------------------------------------

def compute_reward(position: torch.Tensor,
                         y_true: torch.Tensor,
                         vol: torch.Tensor,
                         yhat_conf: torch.Tensor,
                         scale: float = 1000.0,
                         alpha_mag: float = 10.0,
                         lambda_vol: float = 0.0,
                         lambda_turn: float = 0.0):
    """
    position: [B] in [0,2]
    y_true: [B] real returns (can be small)
    vol: [B] recent volatility
    yhat_conf: [B] confidence scalar in [0, inf) (we'll squash with sigmoid)
    scale: multiply reward to amplify signal
    alpha_mag: weight for |y| when combining direction and magnitude
    lambda_vol: penalty coefficient for volatility risk (multiplied by (position*vol)^2)
    lambda_turn: penalty coefficient for turnover (use abs(position) as proxy)
    """

    # direction: 1 if position>0 else 0
    dir_taken = (position > 0).float()

    # sign of real return
    sign_y = torch.sign(y_true)

    # base direction reward: +1 if long and y>0, -1 if long and y<0, 0 if not long
    reward_dir = dir_taken * sign_y  # values in {-1,0,1}

    # magnitude factor: emphasise larger returns (but raw returns are small)
    reward_mag = torch.abs(y_true)  # small
    combined = reward_dir * (1.0 + alpha_mag * reward_mag)

    # confidence scaling (use sigmoid to map to approx [0,1])
    conf = torch.sigmoid(5.0 * yhat_conf)  # tuned factor 5.0
    combined = combined * (0.5 + conf)  # keep some base signal

    # scale up
    combined = combined * scale

    # risk penalty
    risk_pen = lambda_vol * (position * vol)**2

    # turnover penalty: as proxy, penalize large positions (since we don't track prev_action per index reliably here)
    turn_pen = lambda_turn * torch.abs(position)

    reward = combined - risk_pen - turn_pen
    return reward

# -----------------------------------------------------------------------------
# 2. 数据集定义
# -----------------------------------------------------------------------------
class GRPODataset(Dataset):
    def __init__(self, y_hat_path: str, raw_csv_path: str, window_size: int = 20):
        # 加载 OOS 信号
        self.signals = pl.read_parquet(y_hat_path)
        
        # 加载原始数据用于计算统计量和 reward
        # 假设 raw_csv 包含: ds, y (真实收益), 以及原始价格用于算 vol
        # 这里简化：假设 train.csv 里 y 就是收益
        self.raw = pl.read_csv(raw_csv_path).rename({"market_forward_excess_returns": "y"})
        self.raw = self.raw.with_columns(pl.col("date_id").cast(pl.Int64).alias("ds"))
        
        # Join 信号与原始数据
        self.df = self.raw.join(self.signals, on="ds", how="inner").sort("ds")
        
        self.window_size = window_size
        
        # 预计算滚动统计量 (作为 Policy 输入的一部分)
        # Volatility: 过去 N 天 y 的标准差
        self.df = self.df.with_columns([
            pl.col("y").rolling_std(window_size).alias("rolling_vol").fill_null(0.0),
            pl.col("y").rolling_mean(window_size).alias("rolling_trend").fill_null(0.0),
            pl.col("y_hat").rolling_mean(window_size).alias("yhat_ema").fill_null(0.0),
            pl.col("y_hat").rolling_std(window_size).alias("yhat_std").fill_null(0.0),
                ((pl.col("y_hat") - pl.col("y_hat").rolling_mean(self.window_size))
        / (pl.col("y_hat").rolling_std(self.window_size) + 1e-6)
     ).alias("y_hatz").fill_null(0.0)

        ])
        self.df = self.df.with_columns([
            ((pl.col("y_hat") - pl.col("yhat_ema")) / (pl.col("yhat_std") + 1e-9)).alias("y_hatz"),
            pl.col("y_hatz").abs().alias("conf_abs").fill_null(0.0)
        ])
        # 转换为 numpy 以便快速索引
        self.data = self.df.select([
            "y_hat",       # SFT 预测
            "rolling_vol", # 市场状态 1
            "rolling_trend", # 市场状态 2
            'y_hatz',
            'yhat_ema',
            'yhat_std',
            "conf_abs",
            "y"            # 真实收益 (用于计算 Reward)
        ]).to_numpy()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Input State: [y_hat, vol, trend]
        state = self.data[idx, :7].astype(np.float32)
        # Target info for reward: [y_true, vol]
        # 注意：Reward 取决于当天的真实收益 y
        target_info = self.data[idx, [7,6,1]].astype(np.float32) 
        
        return state, target_info

# -----------------------------------------------------------------------------
# 3. 训练脚本
# -----------------------------------------------------------------------------
def train_grpo():
    artifacts = ArtifactPaths()
    signals_path = Path(artifacts.models_dir) / "sft_y_hat_oos.parquet"
    raw_data_path = Path("kaggle/input/hull-tactical-market-prediction/train.csv")
    print(raw_data_path)
    if not signals_path.exists():
        print(f"Signals file not found: {signals_path}")
        print("Please run kaggle/work/gen_sft_signals.py first.")
        return

    # 配置
    EPOCHS = 10
    BATCH_SIZE = 256
    LR = 1e-3
    
    # 准备数据
    dataset = GRPODataset(str(signals_path), str(raw_data_path))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    # Input dim = 3 (y_hat, vol, trend，y_hatz,)
    policy = PolicyHead(input_dim=7, hidden_dim=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    print("Starting GRPO Training...")
    
    for epoch in range(EPOCHS):
        total_reward = 0
        total_loss = 0
        
        for states, targets in dataloader:
            # states: [B, 3]
            # targets: [B, 2] -> y_true, vol
            # 1. 采样动作
            actions = policy.get_action(states, deterministic=False) # [B]  y_hat
            print(actions)
            # 2. 计算奖励
            y_true = targets[:, 0]
            conf = targets[:,1]
            vol = targets[:,2]
            
            
            rewards = compute_reward(actions, y_true, vol,conf,
                scale=1000.0,
                alpha_mag=10.0,
                lambda_vol=0.0,
                lambda_turn=0.01) # [B]
            
            # 3. 计算 Loss (Policy Gradient / REINFORCE)
            # Loss = - log_prob(a) * (R - baseline)
            # 简单起见用 batch 均值做 baseline
            baseline = rewards.mean()
            adv = rewards - baseline
            
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            # 4. 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
        avg_rew = total_reward / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Reward: {avg_rew:.4f} | Loss: {total_loss:.4f}")
        
    # 保存
    save_path = Path(artifacts.models_dir) / "policy_head.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Policy Head saved to {save_path}")

if __name__ == "__main__":
    train_grpo()

