import numpy as np
import matplotlib.pyplot as plt

# Example data
models = ['VAE (Diagonal)', 'VAE (Full Covariance)', 'VLAE']
runs = ['Run 1', 'Run 2', 'Run 3']

values = np.array([
    [1040, 1055, 1059],
    [1094, 1099, 1103],
    [1148, 1156, 1150],
])

# '#4C72B0', '#5e4c5f', '#999999', '#ffbb6f'
# colors = ['#DD8452', '#8C8C8C', '#F2B701']
colors = ['#5e4c5f', '#999999', '#ffbb6f']

x = np.arange(len(models))

bar_width = 0.13         # bar thickness
spacing = 0.10            # distance between bar centers (slightly larger)
offsets = np.linspace(
    -1.4 * spacing, 1.4 * spacing, len(runs)
)
# -----------------------------------

plt.figure(figsize=(7, 4))

for i, run in enumerate(runs):
    plt.bar(
        x + offsets[i],
        values[:, i],
        width=bar_width,
        color=colors[i],
        label=run
    )

plt.xticks(x, models)
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood Results on MNIST\n(Gaussian output)', pad=12)

# Horizontal grid only
plt.grid(axis='y', color='#E6E6E6', linewidth=1)
plt.gca().set_axisbelow(True)

# Clean spines
for spine in ['top', 'right', 'left']:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines['bottom'].set_color('#CCCCCC')

# Legend at bottom center
plt.legend(
    ncol=4,
    frameon=False,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12)
)

plt.tight_layout()

means = values.mean(axis=1)
stds  = values.std(axis=1, ddof=1)

for i, (mean, std) in enumerate(zip(means, stds)):
    color = '#4C72B0' if i in [0, 2] else 'black'  # blue for VAE (Diagonal) and VLAE
    plt.text(
        x[i],
        mean + std + 3,          # vertical offset (tune if needed)
        f"{mean:.1f} ± {std:.1f}",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
        color=color
    )


plt.savefig("out_images/vae_full_cov_comparision.png", dpi=150)
plt.close()
