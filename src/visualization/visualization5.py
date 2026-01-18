import matplotlib.pyplot as plt
import numpy as np


# Data
step_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
mean_log_likelihood = [1129.3, 1152.0, 1151.3, 819.3, 877.7]
std_log_likelihood = [2.1, 2.6, 4.2, 17.6, 16.5]

x = np.arange(len(step_sizes))


# Figure setup
plt.figure(figsize=(7, 4))

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
grid_color = '#B0B0B0'

bars = plt.bar(
    x,
    mean_log_likelihood,
    width=0.36,                 # slightly wider → reduced gaps
    yerr=std_log_likelihood,
    capsize=4,
    color=colors,
    edgecolor='none'
)


# Axes labels and ticks
plt.xlabel(r"Step Size $\alpha_0$", fontsize=10)
plt.ylabel("Log-Likelihood", fontsize=10)

plt.xticks(x, step_sizes, fontsize=11)
plt.yticks(fontsize=11)


# Grid (horizontal only)
plt.grid(
    axis='y',
    linestyle='--',
    linewidth=0.05,
    alpha=0.7,
    color=grid_color
)


# Axis borders (all visible, grid-matched)
ax = plt.gca()
for side in ['left', 'bottom', 'top', 'right']:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(grid_color)
    ax.spines[side].set_linewidth(0.8)


# Mean ± Std annotation
for i, (mean, std) in enumerate(zip(mean_log_likelihood, std_log_likelihood)):
    plt.text(
        x[i],
        mean + std + 0.25 * std + 20,          # offset above error bar
        f"{mean:.1f} ± {std:.1f}",
        ha='center',
        va='bottom',
        fontsize=9
    )


# Layout and export
ax = plt.gca()

y_top = max(
    m + s for m, s in zip(mean_log_likelihood, std_log_likelihood)
)

ax.set_ylim(top=y_top + 150)  # increase if text still touches border


plt.tight_layout()
plt.savefig(
    "out_images/log_likelihood_vs_step_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
