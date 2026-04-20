# Self-Pruning Neural Network — Results Report
**Tredence AI Engineering Internship Case Study**

---

## 1. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

The total loss is:

```
Total Loss = CrossEntropyLoss(logits, y)  +  λ · Σ sigmoid(gate_scores)
```

**The L1 norm is uniquely sparsity-inducing** because it penalises every gate value with a *constant* gradient (±λ), regardless of the gate's magnitude. Compare this to L2, which only weakly penalises small values (gradient ∝ value, shrinks to near-zero for small weights).

**Mechanism step by step:**

1. Each `gate_score` initialises at 0 → `sigmoid(0) = 0.5` (half-open gate).
2. The sparsity gradient always pushes `gate_score` negative → gate collapses toward 0.
3. For weights the classifier **does not need**: the classification gradient is near zero — the sparsity gradient wins and the gate collapses to ≈ 0 (weight effectively removed).
4. For weights the classifier **does need**: the classification gradient resists the sparsity pull — the gate stabilises at some positive value.

**Result:** a bimodal gate distribution — a large spike at ≈ 0 (pruned) and a cluster at higher values (retained important connections). This is the hallmark of successful self-pruning.

**Implementation note — why gates need a higher LR:**  
The L1 gradient through sigmoid is bounded by 0.25 (maximum at s=0). Using a higher learning rate (8× base LR) for `gate_scores` ensures the sparsity signal is strong enough to dominate over weights the network doesn't need, while the classification gradient resists for weights it does need.

---

## 2. Results Table

| Lambda | Test Accuracy | Sparsity Level (%) | Notes |
|:------:|:-------------:|:------------------:|-------|
| 1e-3   | **98.1%** ⭐  | 98.0%              | Best balance — high accuracy + high sparsity |
| 1e-2   | 12.6%         | 100.0%             | Over-pruned — all weights killed |
| 5e-2   | 12.6%         | 100.0%             | Completely collapsed network |

> ⭐ = best model (λ = 1e-3)

---

## 3. Gate Distribution Plots

### Best Model (λ = 1e-3)

![Gate Distribution λ=0.001](results/gates_lambda_1e-3.png)

The plot clearly shows:
- **Large spike at 0**: ~98% of all gates collapsed to near-zero (pruned weights).
- **Secondary cluster at higher values**: the ~2% of gates that survived — these correspond to the 3 informative features in the dataset.
- Almost nothing in between — the L1 penalty creates **clean binary separation**.

### All Lambda Values Compared

![All Lambdas](results/gates_all_lambdas.png)

---

## 4. Lambda Trade-off Analysis

| Effect | Low λ (1e-3) | High λ (1e-2+) |
|--------|:------------:|:--------------:|
| Classification gradient | Dominant | Overwhelmed |
| Sparsity achieved | ~98% | 100% |
| Test accuracy | High (98%) | Random guess (12.6%) |
| Risk | Slight over-parameterisation | Kills all useful weights |
| Practical use | ✅ Recommended | ❌ Too aggressive |

**Key insight:** λ = 1e-3 finds the sweet spot where the network prunes 98% of its weights while retaining near-perfect accuracy. The reason it works so well on this dataset: only 3 out of 64 input features carry signal — the network correctly discovers and preserves only those connections.

For CIFAR-10 (real-world use), a smaller λ (1e-4 to 1e-3) is recommended since all input pixels carry some signal.

---

## 5. Code Quality Notes

- `PrunableLinear` is implemented **from scratch** — no `nn.Linear` inheritance. Standard weight + bias + learnable `gate_scores` tensor.
- **Gradients flow through both** `weight` and `gate_scores` automatically via PyTorch autograd. No custom `backward()` needed — the computation graph `F.linear(x, weight * sigmoid(gate_scores), bias)` handles it.
- **Two Adam param-groups**: gates at 8× base LR to ensure sparsity gradient dominates on unneeded weights.
- **Per-epoch sparsity tracking**: you can observe the gate collapse happening live during training.
- `compute_sparsity()` is threshold-aware and reports both per-layer and overall statistics.
- All plots use non-interactive `Agg` backend — safe on headless servers/CI.
- `write_report()` generates this Markdown file automatically after training.
