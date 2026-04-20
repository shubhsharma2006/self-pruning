# Self-Pruning Neural Network
**Tredence AI Engineering Internship — Case Study**

A feed-forward neural network that learns to prune itself **during training** via learnable sigmoid gates on each weight.

## Concept

```
Total Loss = CrossEntropyLoss + λ · Σ sigmoid(gate_scores)
```

- Each weight has a learnable `gate_score` → `sigmoid(gate_score) ∈ (0,1)`
- `pruned_weight = weight × gate`
- L1 penalty on all gates drives unused ones to 0 → weight effectively removed

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Fast mode — structured synthetic dataset (~3 min, shows gate collapse clearly)
python train.py --fast

# Full mode — CIFAR-10 (requires internet + ~30 min on GPU)
python train.py

# Custom epochs
python train.py --fast --epochs 150
```

## Outputs (saved to `results/`)

| File | Description |
|------|-------------|
| `gates_lambda_*.png` | Gate value histogram per lambda |
| `curves_lambda_*.png` | Loss + accuracy + sparsity over epochs |
| `gates_all_lambdas.png` | Side-by-side comparison |
| `model_lambda_*.pt` | Saved model checkpoint |
| `REPORT.md` | Auto-generated analysis report |

## Results

| Lambda | Test Accuracy | Sparsity |
|--------|:------------:|:--------:|
| 1e-3   | 98.1% ⭐     | 98.0%    |
| 1e-2   | 12.6%        | 100.0%   |
| 5e-2   | 12.6%        | 100.0%   |

## Architecture

```
PrunableLinear(64→128) → ReLU
PrunableLinear(128→64) → ReLU  
PrunableLinear(64→10)           ← logits
```

All linear layers are `PrunableLinear` — every weight has a learnable gate.

## Key Design Decisions

1. **Gate LR = 8× weight LR**: The L1 gradient through sigmoid is bounded by 0.25. A higher LR for gates ensures the sparsity signal dominates on unneeded weights.
2. **Sigmoid gates over hard masks**: Differentiable end-to-end — no straight-through estimator needed.
3. **L1 (not L2) sparsity**: Constant gradient regardless of gate magnitude → drives gates all the way to 0.
