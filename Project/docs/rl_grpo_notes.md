## RL & GRPO Learning Notes (Organized by your thought process)

This note is not a formal paper derivation, but a path **from intuition to formula to code**, following your questions, connecting concepts to help you review later.

---

## 1. Starting from our Project: What are we actually doing?

Let's start with the statement you fully understood:

- **Our current trading RL setup is effectively a "Single-Step Trajectory"**:
  - Each day is treated as a trajectory:
    `state_t` (Today's market state + SFT predictors)
    → `action_t` (Today's position 0–2)
    → `reward_t` (Today's actual PnL + Risk penalty)
  - So a trajectory has only 1 time step t.
  - This is why you don't see an explicit sum over time t in the loss code.

The loss function in code looks like this (conceptual):

```python
loss = -(log_probs * advantages).mean()
```

- `log_probs`: `log πθ(a_t | s_t)` for each sample (day)
- `advantages`: Advantage `A_t` for each sample
- `.mean()`: Average over the entire batch = Approximating the expectation `E[...]` using this batch of samples

You have realized:

- **If the trajectory has only 1 step**:
  The mathematical `Σ_t` degenerates to 1 term;
  Only the average over "different trajectories" (different days) remains.

---

## 2. The two numbers output by MLP: What are alpha / beta exactly?

Your initial confusion:

- "Don't we just have two actions, alpha and beta?"

Clarification:

- **alpha / beta are NOT actions, they are "Distribution Shape Parameters"**.
- What our PolicyHead does:
  1. Look at state, output two real numbers `raw_alpha`, `raw_beta`
  2. Pass through Softplus to become positive, then +1 to get `alpha`, `beta`
  3. Use them to define a **Beta Distribution**: `dist = Beta(alpha, beta)`
  4. **During training**, sample from it `sample ~ Beta(...)`, and map to `[0, 2]`:
     ```python
     sample = dist.sample()      # Random number in [0,1]
     action = 2.0 * sample       # Scale to [0,2]
     ```

Analogy:

- **alpha / beta**: The shape of the dice (Fair dice? Biased to one side?)
- **action**: The point rolled this time

We want a "Distribution" instead of "A fixed value" because:

- **Training**: Need randomness (exploration) to try different positions;
- **Inference**: Use the mean (expectation) of the distribution for a stable deterministic decision.

---

## 3. Trajectory, Time Step, Action: What is Σ summing over?

Your original understanding (very close to truth):

- A trajectory has many actions;
- The loss should be "sum of log_prob of each action × its advantage";
- Finally average over trajectories.

The mathematical notation of standard RL is exactly this:

```text
J(θ) ≈ (1/N) Σ_{i=1..N} Σ_{t=1..T_i} [ log πθ(a_{i,t} | s_{i,t}) * A_{i,t} ]
```

- Outer Σ_i: Average over **different trajectories τ_i** (N trajectories)
- Inner Σ_t: Sum contributions of **actions at different times t** within the same trajectory

In our project:

- We **artificially simplify to "Single-Step Trajectory"**:
  One trajectory per day, trajectory length T=1.
- So `Σ_t` has only 1 term remaining, and the code looks like:

```python
loss = -(log_probs * advantages).mean()
```

Here `.mean()`:

- Is averaging over all "trajectory i" (daily samples);
- Mathematically corresponds to `(1/N) Σ_i`, the inner Σ_t is omitted because T=1.

You summarized it well:

> Monte Carlo is actually sampling multiple trajectories, then the summation is weighted advantage sum over multiple actions in one trajectory, and finally uniform average over all trajectories.

Here: **A trajectory has only 1 action**, so only the last averaging step remains.

> Extra Tip: In **LLM scenarios**, a prompt (state_0) usually samples multiple complete trajectories (different responses), and each trajectory contains multiple actions (token-level decisions).
> So a batch might have B prompts, each prompt samples N trajectories, total B×N trajectories, and each trajectory sums over token dimension.
> Our current trading project is single-step trajectory, so "sample one action per state" is sufficient to build Monte Carlo estimate; no need to repeatedly sample multiple trajectories for the same prompt like LLM. The abstract math framework is the same, just different task structures.

---

## 4. Expectation & Monte Carlo: E[...] in paper vs .mean() in code

Papers often write:

```text
J(θ) = E_{τ ~ πθ}[ R(τ) ]
∇J(θ) = E_{τ ~ πθ}[ Σ_t log πθ(a_t | s_t) * A_t ]
```

You grasped this point:

- The outer `E[...]` is "Expectation over all possible trajectories"
- In reality, we cannot enumerate all trajectories, we can only:
  - Sample N trajectories (or N samples)
  - Use sample average to **approximate expectation**.

This is the concrete meaning of "Monte Carlo Estimate" in RL:

```python
loss = -(log_probs * advantages).mean()
```

- `log_probs * advantages`: Corresponds to `log π * A` for a single sample/step
- `.mean()`: Average over the batch (collection of samples), approximating `E[...]`

You summarized the essence accurately later:

> Monte Carlo is actually sampling multiple trajectories, then doing Σ within a trajectory, then averaging over all trajectories.

---

## 5. Baseline / Advantage: Why subtract "Average Reward"?

Original REINFORCE:

```text
Loss_pg = - E[ log πθ(a|s) * R ]
```

Problem:

- If all R are positive (e.g., all +10), all action probabilities will be pushed up;
- High variance, unstable training.

Solution: Introduce **baseline**, construct Advantage:

```text
A = R - baseline
Loss = - E[ log π(a|s) * A ]
```

Simplest baseline:

- Just the average reward of the current batch `baseline = rewards.mean()`;
- Thus:
  - Better than average (A>0): log_prob is "encouraged" to increase;
  - Worse than average (A<0): log_prob is "punished" to decrease.

You caught the key point:

- For simplest GRPO implementation, **using batch reward mean as baseline is sufficient**,
- No need to introduce independent Value network (Critic), keeping it Actor-only reduces complexity.

This is very suitable for our current project.

---

## 6. What are ratio & clip / KL penalty doing in PPO / GRPO?

You asked:

- Why do many places have KL penalty?
- What's the difference from clip?

Essentially they do the same thing: **Prevent updating too aggressively ("Over-learning")**.

### 6.1 ratio + clip (PPO / GRPO)

PPO style objective:

```text
r(θ) = π_new(a|s) / π_old(a|s)
L_clip(θ) = E[ min( r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A ) ]
```

Explanation:

- If new policy π_new doesn't change much from old policy π_old (r is within [1-ε, 1+ε]), use `r * A` normally;
- If changed too much (r out of range), force clip to `1±ε`;
- Prevents a single gradient step from destroying the policy shape.

### 6.2 KL Penalty

Another common approach is adding KL penalty:

```text
loss = loss_pg + β * KL(π_new || π_old)
```

- Larger KL means larger difference between new and old policies;
- Multiply by weight β to penalize when deviation is too large, forcing updates to be "slower".

**For our current small MLP:**

- Can start without KL / clip,
  Use small learning rate + baseline + appropriate early stopping;
- If training proves unstable later, consider adding a simple KL penalty as the easiest enhancement.

---

## 7. Training "Random Sample" vs Inference "Deterministic Action"

Your initial confusion:

- "Since we use RL, why random sample? What about online inference?"

Key Difference:

### 7.1 Training Phase (Exploration Mode)

We want the policy to try different actions:

```python
alpha, beta = policy_head(state)
dist = Beta(alpha, beta)
sample = dist.sample()      # Draw one in [0,1]
action = 2.0 * sample       # Position [0,2]
```

- Each training, same state might draw different action;
- Different actions lead to different rewards;
- This allows comparing: **"How much better is action A than action B?"**, and writing this info back to parameters via advantage.

### 7.2 Inference Phase (Exploitation Mode)

Online, we need a stable, repeatable decision:

```python
alpha, beta = policy_head(state)
mean = alpha / (alpha + beta)   # Mean of Beta in [0,1]
action = 2.0 * mean             # Map to [0,2]
```

- No more sampling;
- Behavior degrades to a normal regression model: "Input Vector → Real Number Position".

Conclusion:

- **Excess randomness only appears in training**, for exploration;
- **Inference can be completely deterministic**.

---

## 8. Analogy to Transformer / Generative Models: Reward for whole sentence or each token?

Your question:

- "When LLM generates, is reward for the whole paragraph or each token?"
- "If for whole paragraph, how to apply reward to each token?"

Standard RLHF / GRPO-LLM approach:

1. **Generate whole response**:
   - Given a prompt, model generates complete sequence `y_1,...,y_T`.
2. **Score `R` for the whole sentence** (from reward model or rules).
3. **Distribute this `R` to the log_prob of each token**:

   ```text
   Loss = - Σ_{t=1..T} [ log πθ(y_t | y_<t, x) * R ]
   ```

   - Although R is for the sentence, every token choice contributed to it;
   - So multiply every time step t by the same R, telling the model:
     > "This whole sentence is good, decisions at all these steps should be encouraged."

Your previous thought:

- "We can't enumerate permutations for every word to generate a sentence, right?"

Indeed not, the approach is:

- Use current policy πθ to **sample a few sentences** (trajectories τ);
- Score Reward for each trajectory;
- Use statistics of these samples to approximate theoretical `E_{τ ~ πθ}[·]`;
- No need to enumerate all possible sentences.

This logic is identical to our trading RL, just:

- Trading: A trajectory is "sequence of positions over days";
- Text: A trajectory is "output sequence of tokens".

---

## 9. Summary: Distilling your understanding into "Golden Sentences" for review

1. **Monte Carlo**: Simply "Sample multiple trajectories from current policy, use their average to approximate theoretical expectation E[...]".
2. **Trajectory vs Action**: A trajectory consists of multiple steps (s_t, a_t, r_t), inner Σ_t sums contributions within a trajectory, outer Σ_i / mean averages over multiple trajectories.
3. **alpha / beta**: Not actions, but "personality of action distribution"; real action is sampled from Beta(α,β).
4. **Advantage**: `A = R - baseline`, simplest baseline is batch reward mean; A>0 actions encouraged, A<0 actions suppressed.
5. **Our Project**: Each trajectory has only 1 step (Today State → Today Action → Today Reward), so Σ_t over time degenerates, leaving only average over batch.
6. **Training vs Inference**: Training uses random sample (exploration), inference uses mean (exploitation), externally looks like normal "Input → Position" model.
7. **PPO/GRPO**: Added constraints like ratio / clip / KL on top of basic `log_prob * A`, aiming to "not change too much at once", improving stability.

If you want to match these with specific code later, we can write a "6-line dummy environment + 20-line training loop" toy version in `kaggle/work/grpo.py`, marking every Σ / E[...] on specific tensor dimensions. That way you can switch freely between "Paper Formula → Code Implementation".
