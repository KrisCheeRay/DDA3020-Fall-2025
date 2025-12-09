# A Robust SFT-GRPO Pipeline for Hull Tactical Market Prediction

**Authors:** Yiteng Mao (123090419), Rui Cai (123090007), Yixi Cai (123090010), Zirun Zheng (123090891)  
**Institution:** The Chinese University of Hong Kong, Shenzhen

---

## Abstract

Financial market prediction is characterized by high noise, non-stationarity, and evolving distributions. In this course project for the Kaggle Hull Tactical Market Prediction competition, we propose a two-stage end-to-end framework: **Supervised Fine-Tuning (SFT)** followed by **Generalized Reinforcement Policy Optimization (GRPO)**. Unlike traditional approaches that rely solely on regression loss, we decouple signal generation from decision-making. First, we perform rigorous feature engineering to select stable predictive factors. Second, we conducted a comparative analysis between Transformer-based (PatchTST) and MLP-based (N-HiTS) models, ultimately selecting N-HiTS for its superior capability to capture strong negative correlations in key factors. We generate "Rotten Apple" OOS signals to simulate realistic errors. Finally, we train a Policy Network (MLP) using Reinforcement Learning to optimize a comprehensive reward function ($PnL - Risk - Turnover$) based on these signals and market states.

---

## 1. Introduction

Predicting stock market returns is a notoriously difficult task due to the low signal-to-noise ratio. The Hull Tactical Market Prediction competition challenges participants to forecast future returns using a provided set of proprietary features. Traditional supervised learning often struggles to translate prediction accuracy (e.g., MSE) directly into profitability (PnL) due to the disconnect between loss functions and financial objectives.

To address this, we developed a pipeline that separates *signal generation* from *decision making*. We treat the base model's predictions not as the final answer, but as a feature state for a Reinforcement Learning agent. This allows the agent to learn a policy that adapts to the base model's uncertainty, balancing risk and return dynamically.

---

## 2. Data Exploration and Feature Engineering

The dataset consists of anonymized financial features. Our goal was to identify features with high predictive power and low correlation to ensure model robustness.

### 2.1 Evaluation Metrics
We evaluated features using three key metrics:
1.  **Information Coefficient (IC) & Rank IC**: The correlation between the feature value and the forward return over a rolling window. A stable, high IC indicates predictive power.
2.  **Information Ratio (IR)**: The ratio of the mean IC to the standard deviation of the IC, measuring the stability of the factor.
3.  **Grouped Returns**: We divided time periods into deciles based on feature values. A monotonic relationship between feature deciles and returns indicates strong discriminative ability.

### 2.2 Analysis and Selection

Our analysis revealed the presence of **"Super Factors"**. Specifically, factors `E2` and `E3` exhibited a Rank IC of approximately **-0.15** with a negative Information Ratio (IR) exceeding **-2.3**. In quantitative finance, a Rank IC of 0.05 is typically considered strong; -0.15 indicates an exceptionally strong and stable negative linear relationship with returns.

*Figure 1: (Left) Cumulative IC and Rank IC for feature E2. (Right) Grouped cumulative returns for feature E3.*
![Feature Analysis](Analysis/e2_cic.png) ![Grouped Returns](Analysis/e3_qcr.png)

As shown above, the Cumulative IC for feature `E2` approximates a straight line with a negative slope, confirming this stable negative correlation. Similarly, `E3` exhibits excellent monotonicity in grouped returns.

Initial screening selected 21 promising features. To reduce redundancy, we computed the correlation matrix. For pairs with correlation $> 0.7$, we retained the one with higher IR.

*Figure 2: Correlation matrix of the final selected 8 features.*
![Correlation Matrix](Analysis/2_2_f_corr_mat.png)

The final selected set consists of 8 features: `["E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"]`. These features serve as the input for both our SFT models and the RL Policy Network. We use a `FeatureStore` to ensure consistent `StandardScaler` transformation between training and inference.

---

## 3. Methodology

Our architecture follows a Sim-to-Real philosophy, ensuring the training process mimics the online inference environment.

### 3.1 Stage 1: Supervised Fine-Tuning (SFT)

#### Model Selection: The PatchTST vs. N-HiTS Dilemma
In the initial phase, we experimented with two state-of-the-art time-series architectures: **PatchTST** (Transformer-based) and **N-HiTS** (MLP-based). Our empirical results strongly favored N-HiTS.

| Model | MSE | IC (Corr) | Hit Rate |
| :--- | :--- | :--- | :--- |
| PatchTST | 0.00007963 | -0.2466 | 50.00% |
| **N-HiTS** | **0.00006249** | **0.0611** | **70.00%** |
| Ensemble | 0.00005537 | -0.1082 | 60.00% |

**Analysis of Failure (PatchTST):**
PatchTST employs a Channel Independence (CI) strategy, treating each variable as an isolated univariate series. However, our data analysis showed that returns are driven by strong **cross-sectional factors** (E2, E3). PatchTST fails to model the explicit relationship $y \approx w \cdot E2 + b$ because it only looks at the history of $E2$ to predict $E2$, rather than using $E2$ to predict $y$. Consequently, it achieved a Hit Rate of only 50% (random guess) and a negative IC, indicating overfitting to noise.

**Analysis of Success (N-HiTS):**
N-HiTS, despite being a time-series model, effectively functions as a stacked MLP that can perform multivariate regression. It successfully captured the strong linear signal from the "Super Factors" E2 and E3. The high Hit Rate (70%) and positive IC confirm that N-HiTS learned the correct directional relationship, leveraging the strong alpha present in the selected features. Therefore, we adopted an "All-in N-HiTS" strategy.

#### The "Rotten Apple" Mechanism
To train the downstream RL agent effectively, we cannot feed it "perfect" predictions trained on the full dataset (which would cause look-ahead bias). Instead, we need signals that reflect the model's true out-of-sample errors. We implemented a **Rolling Cross-Validation** mechanism, which we term "Rotten Apples":
$$
\hat{y}_{t} = f_{\theta_{t-k}}(\mathbf{x}_{t})
$$
where the model $f$ used to predict at time $t$ is trained only on data up to $t-k$. This prevents look-ahead bias and provides the RL agent with realistic, noisy signals ($\hat{y}$) alongside the ground truth ($y$).

### 3.2 Stage 2: Generalized Reinforcement Policy Optimization (GRPO)
We treat trading as a single-step trajectory RL problem. The agent observes a state $s_t$ and outputs an action $a_t \in [0, 2]$ (Position).

**State Space ($s_t \in \mathbb{R}^{11}$):**
*   Signal: $\hat{y}_{t}$ (prediction from N-HiTS)
*   Market State: Rolling Volatility, Rolling Trend (calculated from past returns)
*   Features: The 8 selected raw features

**Crucially, we explicitly include the raw features (especially E2, E3) in the state.** This allows the Policy Network to "see" the strong factors directly, enabling it to override the N-HiTS signal if necessary or learn a non-linear risk adjustment based on factor regimes.

**Policy Network (Actor):**
A Multi-Layer Perceptron (MLP) maps the state to parameters $\alpha, \beta$ of a Beta distribution.
$$
\alpha, \beta = \text{Softplus}(\text{MLP}(s_t)) + 1
$$
During training, we sample action $a_t \sim \text{Beta}(\alpha, \beta)$ scaled to $[0, 2]$ to encourage exploration. During inference, we use the mean of the distribution for deterministic execution.

**Reward Function:**
We designed a composite reward to balance profit and risk:
$$
R_t = \text{PnL}_t - \lambda_{risk} \cdot (a_t \cdot \sigma_t)^2 - \lambda_{turnover} \cdot |a_t - a_{t-1}| \cdot \text{Cost} + \lambda_{dir} \cdot \mathbb{I}(\text{sign}(\hat{y}) = \text{sign}(y))
$$
where PnL is amplified to facilitate gradient flow. We used Gradient Clipping and Learning Rate Scheduling to stabilize training.

---

## 4. Evaluation

### 4.1 Experimental Setup
The submission was evaluated using the Kaggle Evaluation API.
*   **Environment:** Kaggle Notebook (Offline). We utilized a custom offline package installation mechanism for `neuralforecast`.
*   **Cold Start:** A momentum-based heuristic ($\tanh(return \times 50)$) is used when the history window ($<60$ days) is insufficient for the N-HiTS model.
*   **Inference:** Rolling window prediction. For each batch, the new data is appended to a history buffer to maintain the context required by the time-series model.

### 4.2 Results
**Local Validation:** Our best policy achieved a significantly lower loss on the validation set compared to the baseline SFT signals, indicating the agent learned to temper bets during high volatility or uncertain market conditions.

While the absolute score suggests room for improvement, the pipeline functioned correctly without runtime errors, verifying the robustness of our engineering implementation.

---

## 5. Conclusion

We successfully implemented an end-to-end SFT-GRPO trading pipeline. By decoupling signal generation from portfolio decision-making, we created a flexible system capable of optimizing for complex financial objectives.

**Key Takeaways:**
1.  **Factor Analysis Drives Architecture:** The discovery of super-strong negative linear factors (E2/E3 with Rank IC -0.15) directly informed our decision to abandon PatchTST in favor of N-HiTS, which could better model these relationships.
2.  **Honest Validation:** Generating OOS signals ("Rotten Apples") is essential for training effective RL agents.
3.  **Infrastructure:** Robust handling of online inference constraints (cold start, package management) is as important as the model itself.

---

## 🚀 How to Reproduce

To reproduce our results locally or deploy to Kaggle, follow these steps:

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/RichardCyx/Hull-Tactical---Market-Prediction.git
cd Hull-Tactical---Market-Prediction
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure the competition data is in `kaggle/input/hull-tactical-market-prediction/`:
*   `train.csv` (or `train_feature_selected.csv`)
*   `test.csv`

### 3. Training Pipeline
Run the following scripts in order:

**Step 1: Train Base Model (SFT)**
```bash
python kaggle/work/sft_nhits.py
```
*Outputs: `models/nhits_checkpoints/`*

**Step 2: Generate OOS Signals ("Rotten Apples")**
```bash
python kaggle/work/gen_sft_signals.py
```
*Outputs: `models/sft_y_hat_oos.parquet`*

**Step 3: Train Policy Agent (GRPO)**
```bash
python kaggle/work/train_grpo.py
```
*Outputs: `models/policy_head_best.pt`*

### 4. Local Inference Test
Simulate the Kaggle online environment locally:
```bash
python kaggle/work/starter_notebook.py
```
*Outputs: `submission.parquet`*

### 5. Files for Kaggle Submission
Upload the following files to a private Kaggle Dataset:
*   `models/scaler.pkl`
*   `models/features.json`
*   `models/nhits_checkpoints.zip` (Zip the directory)
*   `models/policy_head_best.pt`
*   `neuralforecast` wheel files (for offline install)

Copy the content of `kaggle/work/starter_notebook.py` to your Kaggle Notebook and run!
