from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# logdirs = [
#     (
#         "model0.3",
#         "checkpoints/MNIST/VLAE/2025-12-19_15:48:58.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0"
#     ),
#     (
#         "model0.5",
#         "checkpoints/MNIST/VLAE/2025-12-20_16:08:01.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0"
#     )
# ]

logdirs = [
    (
        "model0.1",
        "checkpoints/MNIST/VLAE/2025-12-13_15:43:39.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0"
    ),
    (
        "model0.7",
        "checkpoints/MNIST/VLAE/2025-12-22_06:03:40.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0"
    ),
    (
        "model1.0",
        "checkpoints/MNIST/VLAE/2025-12-23_22:25:45.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0"
    )
]
out_dir = "out_images"

def load_scalar(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def value_at_step(steps, values, step):
    return np.interp(step, steps, values)

model_data = []
y_min, y_max = np.inf, -np.inf

for name, logdir in logdirs:
    train_steps, train_loss = load_scalar(logdir, "train/loss")
    test_steps, test_loss = load_scalar(logdir, "test/loss")
    ll_steps, test_ll = load_scalar(logdir, "test/loglikelihood")

    best_idx = np.argmax(test_ll)
    best_step = ll_steps[best_idx]
    best_ll = test_ll[best_idx]

    best_train_loss = value_at_step(train_steps, train_loss, best_step)
    best_test_loss  = value_at_step(test_steps, test_loss, best_step)

    y_min = min(y_min, min(train_loss), min(test_loss))
    y_max = max(y_max, max(train_loss), max(test_loss))

    model_data.append(
        dict(
            name=name,
            train_steps=train_steps,
            train_loss=train_loss,
            test_steps=test_steps,
            test_loss=test_loss,
            best_step=best_step,
            best_ll=best_ll,
            best_train_loss=best_train_loss,
            best_test_loss=best_test_loss,
        )
    )

for data in model_data:
    plt.figure(figsize=(9, 6))

    plt.plot(data["train_steps"], data["train_loss"], label="Train Loss")
    plt.plot(data["test_steps"], data["test_loss"], label="Test Loss")

    plt.axvline(
        data["best_step"],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Best Epoch"
    )

    plt.scatter(data["best_step"], data["best_test_loss"], zorder=3, color="black")

    plt.annotate(
        f"best epoch={data['best_step']}\nlog-likelihood={data['best_ll']:.1f}",
        (data["best_step"], data["best_test_loss"]),
        textcoords="offset points",
        xytext=(8, 16),
        bbox=dict(boxstyle="round", fc="none", alpha=0.9)
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train vs Test Loss")
    plt.ylim(y_min, y_max)   # shared scaling
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{out_dir}/train_vs_test_loss_{data['name']}.png", dpi=150)
    plt.close()
