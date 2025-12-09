# Project Structure

## Core Files

### Kaggle Submission
- `kaggle/work/starter_notebook.py` - **Main submission notebook** (GRPO-based strategy)
- `kaggle/work/online_learning_simple.py` - **Best performing model** (Simple MLP with online learning, Score: 2.6)

### Training Scripts
- `kaggle/work/train_grpo.py` - GRPO policy head training
- `kaggle/work/gen_sft_signals.py` - Generate out-of-sample signals for GRPO training

### Baseline/Experimental Scripts
- `kaggle/work/baseline_constant_1.py` - Baseline: always output 1.0
- `kaggle/work/baseline_random_biased.py` - Baseline: random [0.7-1.3] with bullish bias

### Source Code
- `src/configs.py` - Configuration classes (DataSchema, ArtifactPaths)
- `src/feature_store.py` - Feature preprocessing and standardization
- `src/policy_head.py` - Policy Head V1 (Beta distribution)
- `src/policy_head_v2.py` - Policy Head V2 (Bernoulli + Beta hybrid)

### Models & Artifacts
- `models/` - Trained models and artifacts (excluded from git, see .gitignore)
  - `nhits_checkpoints/` - N-HiTS model weights
  - `policy_head_best.pt` - Best GRPO policy head
  - `scaler.pkl` - Feature scaler
  - `features.json` - Feature metadata

### Documentation
- `README.md` - Main project documentation
- `PROJECT_REPORT.md` - Detailed project report
- `report.tex` - LaTeX report (ICLR 2026 format)
- `docs/` - Additional documentation

### Configuration
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment
- `.gitignore` - Git ignore rules

## Key Results

| Model | Score | Notes |
|-------|-------|-------|
| **Online Learning MLP** | **2.6** | Best performing, simple and effective |
| GRPO V1 | 0.43 | Complex RL-based strategy |
| GRPO V2 | <0.43 | Worse than V1 |
| Constant 1.0 | Baseline | Reference point |

## Quick Start

### For Kaggle Submission
1. Use `kaggle/work/starter_notebook.py` (GRPO) or `online_learning_simple.py` (best score)
2. Ensure models are uploaded to Kaggle Dataset
3. Add neuralforecast wheels as Dataset input

### For Local Training
1. Install dependencies: `pip install -r requirements.txt`
2. Train SFT: `python kaggle/work/gen_sft_signals.py`
3. Train GRPO: `python kaggle/work/train_grpo.py`

## Notes

- Models directory is excluded from git (too large)
- Data files (`kaggle/input/`) are excluded from git
- Temporary outputs (`submission*.parquet`) are excluded
- See `.gitignore` for complete exclusion list

