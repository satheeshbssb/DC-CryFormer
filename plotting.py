import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_confusion_raw(cm, class_names, title, save_path, cmap="Blues"):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=200)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12)
    vmax = cm.max()
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > 0.55 * vmax else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_norm(cm_norm, class_names, title, save_path, cmap="Blues"):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=200)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_results, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=200)
    for c, name in enumerate(class_names):
        r = roc_results[c]
        ax.plot(r["fpr"], r["tpr"], lw=1.8, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, title_prefix: str, save_path: str):
    """history: {"train_acc": [...], "val_acc": [...], "train_loss": [...], "val_loss": [...]}"""
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=200)
    axes[0].plot(epochs, history["train_acc"], label="Training accuracy")
    axes[0].plot(epochs, history["val_acc"], label="Validation accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title_prefix}: Accuracy"); axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Training loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Focal loss")
    axes[1].set_title(f"{title_prefix}: Loss"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam_overlay(spectrogram: np.ndarray, cam_gc: np.ndarray, cam_gcpp: np.ndarray,
                          class_name: str, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=200)
    axes[0].imshow(spectrogram, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title(f"{class_name}: input"); axes[0].axis("off")

    axes[1].imshow(spectrogram, origin="lower", aspect="auto", cmap="gray")
    axes[1].imshow(cam_gc, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM"); axes[1].axis("off")

    axes[2].imshow(spectrogram, origin="lower", aspect="auto", cmap="gray")
    axes[2].imshow(cam_gcpp, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    axes[2].set_title("Grad-CAM++"); axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
