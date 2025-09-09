
# 🔥 AdaptiveOptimizer

A **drop-in wrapper** for any PyTorch (or bitsandbytes) optimizer that adds **loss-driven** and **reward-driven** adaptation to hyperparameters — all *outside the computation graph*.

Instead of building yet another optimizer, you keep using **AdamW, SGD, RMSprop, Adagrad, Adafactor, and their 8-bit variants**, and `AdaptiveOptimizer` will dynamically adjust **learning rate, momentum, betas, alpha, and weight decay** before every `.step()`.

---

## ✨ Key Ideas

* ✅ **Keep your optimizer**: AdamW, SGD, RMSprop, Adagrad, etc. remain intact.
* ✅ **Adapt outside autograd**: no schedulers, no meta-loops, no graph overhead.
* ✅ **Two control channels**:

  * **Loss-driven**: adapt from the training loss curve.
  * **Reward-driven**: adapt from external signals (RL reward, accuracy, human feedback).
* ✅ **Composable rules**: `trend`, `variance`, `cosine_restart`, `relative`, etc.
* ✅ **Safe**: Only touches knobs your optimizer actually has.

---

## 🧠 The Theory

> *An optimizer is split into two layers:*
>
> 1. **Base Optimizer** → applies gradients (e.g. AdamW, SGD).
> 2. **Adaptive Controller** → updates its hyperparameters dynamically from *loss* and/or *reward* signals.

This makes `AdaptiveOptimizer` a **universal controller**:

* Loss provides short-term feedback.
* Reward provides long-term or external feedback.
* The base optimizer remains stable, proven, and unchanged.

---

## ⚡ Installation

Just copy `adaptive_optimizer.py` into your project.
It depends only on **PyTorch** and **NumPy**.

```bash
pip install torch numpy
```

---

## 🚀 Usage

```python
from torch.optim import AdamW
from adaptive_optimizer import AdaptiveOptimizer

# 1) Build a normal optimizer
base_opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 2) Wrap it
opt = AdaptiveOptimizer(
    base_opt,
    lr_rule="trend+variance",
    betas_rule=("relative", "variance"),
    wd_rule="trend",
    reward_rule="trend",
    cycle_length=500,
    cycle_mult=2
)

# 3) Training loop
for step, (x, y) in enumerate(dataloader, 1):
    loss = criterion(model(x), y)
    loss.backward()

    # Loss-based adaptation
    opt.step_loss(loss)

    # Reward-based adaptation (example: 1 / (1+loss))
    reward = 1.0 / (1.0 + loss.item())
    opt.step_reward(reward)

    opt.step()
    opt.zero_grad()

    if step % 100 == 0:
        print(f"[{step}] Loss={loss.item():.4f} | LR={opt.lr:.6e} | WD={opt.wd:.5f}")
```

---

## 🔧 Supported Rules

### Learning Rate (`lr_rule`)

* `"trend"` → increase if loss improves, decay otherwise.
* `"variance"` → shrink with high loss variance.
* `"cosine"` / `"cosine_restart"` → cosine annealing cycles.

### Momentum (`momentum_rule`)

* `"inverse"` → inverse-loss scaling.
* `"relative"` → increase when improving, decrease otherwise.
* `"normalized"` → normalize by loss magnitude.

### Betas (AdamW) (`betas_rule`)

* `"relative"` → beta1 tracks loss improvement.
* `"variance"` → beta2 adapts to loss variance.

### RMSprop Alpha (`alpha_rule`)

* `"stable"` → adjusts based on stability of loss.

### Weight Decay (`wd_rule`)

* `"trend"` → reduce when improving, increase when worsening.

### Reward (`reward_rule`)

* `"scaling"` → scale LR ∝ reward.
* `"trend"` → increase LR/momentum if reward improves.
* `"variance"` → shrink LR if reward variance is high.

---

## 📊 Why Use This?

* **Unifies optimizers**: one wrapper for AdamW, SGD, RMSprop, Adagrad, etc.
* **Dynamic adaptation**: reacts to training dynamics instead of fixed schedules.
* **Reward integration**: bring external signals (RL, accuracy, user feedback) into the optimizer.
* **Lightweight**: \~200 lines, no extra dependencies, no graph overhead.

---

## 📝 License

MIT License. Use freely in research or production.

---

👉 Would you like me to also include a **comparison example** (baseline AdamW vs AdaptiveOptimizer(AdamW)) with plots of loss curves and LR adaptation in the README? That could make the theory very concrete.
