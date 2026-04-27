from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def plot_confusion_matrix(cm, class_names, save_path=None):
    cm_normalized = cm.numpy().astype(float) / (cm.numpy().sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_normalized * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=100
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (%)")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)

