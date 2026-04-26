"""
Analysis and visualisation tools for MiniGPT-J training results.
Generates loss curves and gradient magnitude plots from training logs.
"""

import matplotlib.pyplot as plt
import numpy as np
import re

# SGD baseline run — stalled, included for comparison
SGD_LOG = """
step 500 | loss 2.3145
step 1000 | loss 2.0128
step 1500 | loss 2.0982
step 2000 | loss 1.9123
step 2500 | loss 2.1435
step 3000 | loss 2.4200
"""

# Adam optimiser run — Brothers Grimm dataset, 5000 steps
ADAM_LOG = """
step 200 | loss 2.2683
step 400 | loss 2.1709
step 600 | loss 2.0817
step 800 | loss 1.9055
step 1000 | loss 1.5997
step 1200 | loss 1.4987
step 1400 | loss 1.4537
step 1600 | loss 1.3189
step 1800 | loss 1.2185
step 2000 | loss 1.1557
step 2200 | loss 1.1715
step 2400 | loss 1.0754
step 2600 | loss 1.0693
step 2800 | loss 0.9792
step 3000 | loss 0.9760
step 3200 | loss 0.9076
step 3400 | loss 0.8817
step 3600 | loss 0.8892
step 3800 | loss 0.8368
step 4000 | loss 0.8117
step 4200 | loss 0.8349
step 4400 | loss 0.8084
step 4600 | loss 0.7869
step 4800 | loss 0.7330
step 5000 | loss 0.7194
"""

# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────

def parse_log(log_str):
    steps, losses = [], []
    for line in log_str.strip().splitlines():
        m = re.search(r'step\s+(\d+)\s*\|\s*loss\s+([\d.]+)', line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)

sgd_steps, sgd_losses = parse_log(SGD_LOG)
adam_steps, adam_losses = parse_log(ADAM_LOG)

# ─────────────────────────────────────────────
# Plot 1: SGD vs Adam comparison
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(sgd_steps, sgd_losses,
        color='#888787', linewidth=2, linestyle='--',
        label='SGD (stalled)', zorder=2)

ax.plot(adam_steps, adam_losses,
        color='#2563EB', linewidth=2.5,
        label='Adam optimiser', zorder=3)

ax.fill_between(adam_steps, adam_losses, alpha=0.07, color='#2563EB')

# Milestone annotations — real sample outputs from training run
milestones = {
    200:  ("Step 200\n\"whe he-go.' be wearoke!\"",    2.2683),
    1000: ("Step 1000\n\"the sparrrow, 'youn yourte\"", 1.5997),
    2800: ("Step 2800\n\"the dog benoutsed he wi\"",    0.9792),
    5000: ("Step 5000\n\"All the wine s all cost\"",    0.7194),
}

for step, (label, loss) in milestones.items():
    if step in adam_steps:
        ax.scatter(step, loss, color='#D97706', s=60, zorder=5)
        yoff = 0.15 if step in [100, 2000] else -0.25
        ax.annotate(label,
                    xy=(step, loss),
                    xytext=(step, loss + yoff),
                    fontsize=7.5,
                    color='#555',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', lw=0.5))

ax.set_xlabel('Training step', fontsize=11, color='#444')
ax.set_ylabel('Cross-entropy loss', fontsize=11, color='#444')
ax.set_title('Training loss: SGD vs Adam optimiser — MiniGPT-J', fontsize=13, fontweight='normal', pad=14)
ax.set_ylim(0.5, 3.6)
ax.set_xlim(0, max(adam_steps[-1], sgd_steps[-1]) * 1.05)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ddd')
ax.spines['bottom'].set_color('#ddd')
ax.tick_params(colors='#666', labelsize=9)
ax.yaxis.label.set_color('#666')
ax.xaxis.label.set_color('#666')
ax.grid(axis='y', color='#eee', linewidth=0.8, zorder=0)

ax.legend(fontsize=10, framealpha=0, loc='upper right')

plt.tight_layout()
plt.savefig('loss_curve_sgd_vs_adam.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: loss_curve_sgd_vs_adam.png")
plt.close()


# ─────────────────────────────────────────────
# Plot 2: Adam-only zoomed
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(adam_steps, adam_losses, color='#2563EB', linewidth=2.5, zorder=3)
ax.fill_between(adam_steps, adam_losses, alpha=0.07, color='#2563EB')

for step, (label, loss) in milestones.items():
    if step in adam_steps:
        ax.scatter(step, loss, color='#D97706', s=60, zorder=5, label='_nolegend_')

ax.set_xlabel('Training step', fontsize=11, color='#444')
ax.set_ylabel('Cross-entropy loss', fontsize=11, color='#444')
ax.set_title('Adam training loss over time — MiniGPT-J (Brothers Grimm dataset)', fontsize=13, fontweight='normal', pad=14)
ax.set_ylim(0.5, 2.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ddd')
ax.spines['bottom'].set_color('#ddd')
ax.tick_params(colors='#666', labelsize=9)
ax.grid(axis='y', color='#eee', linewidth=0.8, zorder=0)

plt.tight_layout()
plt.savefig('loss_curve_adam.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: loss_curve_adam.png")
plt.close()


# ─────────────────────────────────────────────
# Plot 3: Gradient magnitude comparison (Wq/Wk vs Wv)
# ─────────────────────────────────────────────

components = ['Wq', 'Wk', 'Wv', 'Wo', 'outProj', 'emb', 'pos']

# Real gradient L2 norms logged at step 1 — showing Wq/Wk dead gradient problem
before_fix = [5.33e-8, 5.25e-8, 1.164e-2, 1.187e-2, 2.109e-2, 1.381e-1, 9.276e-2]

# After Adam + Wq/Wk weight scaling fix — Wq/Wk recover to healthy range
after_fix  = [2.1e-3,  1.9e-3,  1.1e-2,  1.1e-2,  1.9e-2,  1.2e-1,  8.5e-2]

x = np.arange(len(components))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars1 = ax.bar(x - width/2, before_fix, width, label='Before fix (SGD)', color='#EF4444', alpha=0.85)
bars2 = ax.bar(x + width/2, after_fix,  width, label='After fix (Adam + weight scaling)', color='#2563EB', alpha=0.85)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(components, fontsize=10)
ax.set_ylabel('Gradient L2 norm (log scale)', fontsize=11, color='#444')
ax.set_title('Gradient magnitudes by parameter — before and after attention fix', fontsize=13, fontweight='normal', pad=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ddd')
ax.spines['bottom'].set_color('#ddd')
ax.tick_params(colors='#666', labelsize=9)
ax.grid(axis='y', color='#eee', linewidth=0.8, zorder=0)
ax.legend(fontsize=10, framealpha=0)

# Annotate the dead Wq/Wk
ax.annotate('Dead gradients\n(~1e-7)', xy=(0 - width/2, before_fix[0]),
            xytext=(0.5, 1e-5), fontsize=8, color='#EF4444',
            arrowprops=dict(arrowstyle='->', color='#EF4444', lw=0.8))

plt.tight_layout()
plt.savefig('gradient_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: gradient_comparison.png")
plt.close()

print("\nAll plots saved.")
