# Implicit Statistical Reasoning in Transformers: Emergence of Likelihood-Ratio Tests In-Context
## Installation

**Requirements**
- Python **3.13** or higher

We recommend using `uv` for dependency management.

```
bash
pip install uv
uv sync
```

## Tasks

### Task A: Mean Discrimination with Nuisance Shift

Each episode samples:

- a unit vector $\mu \in \mathbb{R}^d$
- a nuisance shift $k \in \mathbb{R}^d$

Data are generated as:

- $y = 1$ : $x \sim \mathcal{N}(\mu + k, I)$
- $y = 0$ : $x \sim \mathcal{N}(-\mu + k, I)$

The Bayes-optimal log-likelihood ratio is linear:

$$
\mathrm{LLR}(x) = 2 \mu^\top (x - k)
$$

This task tests whether the model can:

- infer a latent direction from context
- subtract nuisance shifts
- implement a linear discriminant in-context



### Task B: Variance Discrimination

Each episode samples two variances $\sigma_0, \sigma_1$.

Data are generated as:

- $y = 0$ : $x \sim \mathcal{N}(0, \sigma_0^2 I)$
- $y = 1$ : $x \sim \mathcal{N}(0, \sigma_1^2 I)$

The Bayes-optimal LLR depends on the squared norm:

$$
\text{LLR}(x) = d \log \frac{\sigma_0}{\sigma_1} + \frac{1}{2} \lVert x \rVert^2 \left( \frac{1}{\sigma_0^2} - \frac{1}{\sigma_1^2} \right)
$$

This task tests whether transformers can represent quadratic sufficient statistics and nonlinear decision rules.

## Model Architecture

We use a lightweight **Transformer encoder** trained end-to-end for binary classification.

### Base Architecture

- Context tokens: $(x_i, y_i)$ pairs
- Query token: $x_q$
- Embeddings:
  - `x_proj`: linear projection of inputs 
  - `y_proj`: label embedding
  - learned positional embeddings
- Transformer encoder layers (`nn.TransformerEncoder`)
- Linear readout head producing a single logit

The model predicts the query label using only the context and query, without gradient updates.

## Training Setup

- Episodes generated on-the-fly
- Fixed context size per episode
- Binary classification loss (`BCEWithLogitsLoss`)
- AdamW optimizer with OneCycleLR schedule
- Multiple random seeds for robustness

We evaluate both in-distribution and out-of-distribution settings
(e.g. larger nuisance shifts in Task A).

## Ablation Models

To probe inductive biases and failure modes, we implement several
architectural and data ablations, including:

- **No-label model**: removes access to context labels
- **Shuffled labels**: breaks $x–y$ association
- **Shuffled context**: permutes context order
- **Interleaved embeddings**: separates $x$ and $y$ tokens
- **Frozen attention / frozen QK / frozen positional embeddings**
- **No positional embeddings**
- **Noisy labels**

Each ablation is trained and evaluated identically to the base model.


