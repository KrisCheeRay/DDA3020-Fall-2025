import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # direction head -> Bernoulli p_long
        self.dir_head = nn.Linear(hidden_dim // 2, 1)

        # magnitude head -> Beta params (alpha, beta)
        self.mag_head = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        p_long = torch.sigmoid(self.dir_head(h)).squeeze(-1)  # (B,)
        mag_raw = self.mag_head(h)  # (B,2)
        alpha = F.softplus(mag_raw[:, 0]) + 1e-6
        beta = F.softplus(mag_raw[:, 1]) + 1e-6
        return p_long, alpha, beta

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        """
        Return position in [0,2]:
            direction ~ Bernoulli(p_long)
            magnitude ~ Beta(alpha,beta) scaled * 2
            final position = direction * (2 * magnitude)
        """
        p_long, alpha, beta = self.forward(x)
        if deterministic:
            dir_taken = (p_long > 0.5).float()
            mag_mean = alpha / (alpha + beta)
            mag = mag_mean
        else:
            dir_taken = torch.bernoulli(p_long)
            dist = torch.distributions.Beta(alpha, beta)
            mag = dist.rsample()  # (B,)
        position = dir_taken * (2.0 * mag)
        return  position

    def get_log_prob(self, x: torch.Tensor, action: torch.Tensor):
        """
        Compute log prob of taken action:
        - log_prob_dir: Bernoulli
        - log_prob_mag: Beta (on action/2 in (0,1))
        For action==0 (dir==0) we only count dir log-prob.
        For action>0, we count both.
        Returns: log_prob (B,)
        """
        p_long, alpha, beta = self.forward(x)
        # dir part
        dir_taken = (action > 0).float()
        eps = 1e-8
        log_prob_dir = dir_taken * torch.log(p_long + eps) + (1 - dir_taken) * torch.log(1 - p_long + eps)

        # magnitude part
        mag = (action / 2.0).clamp(eps, 1 - eps)
        dist = torch.distributions.Beta(alpha, beta)
        log_prob_mag = dist.log_prob(mag)  # (B,)

        # combine: only include mag when dir_taken==1
        log_prob = log_prob_dir + dir_taken * log_prob_mag
        return log_prob
