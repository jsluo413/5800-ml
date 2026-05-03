import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)


@torch.no_grad()
def collect_predictions(model, loader, device, pos_label=0):
    model.eval()
    probs_list, preds_list, targets_list = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, pos_label].cpu().numpy()
        probs_list.append(p)
        preds_list.append(logits.argmax(1).cpu().numpy())
        targets_list.append(y.numpy())
    return (np.concatenate(probs_list),
            np.concatenate(preds_list),
            np.concatenate(targets_list))


def compute_metrics(y_true, y_pred, y_prob, pos_label):
    y_bin = (y_true == pos_label).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_bin, y_prob)),
    }


def plot_confusion_matrix(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_training_curves(histories, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        epochs = range(1, len(h["train_acc"]) + 1)
        axes[0].plot(epochs, h["train_loss"], label=f"{name} train", linestyle="--", alpha=0.7)
        axes[0].plot(epochs, h["val_loss"], label=f"{name} val")
        axes[1].plot(epochs, h["train_acc"], label=f"{name} train", linestyle="--", alpha=0.7)
        axes[1].plot(epochs, h["val_acc"], label=f"{name} val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss curves")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy curves")
    for ax in axes:
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_roc(y_true_dict, y_prob_dict, title, out_path, pos_label=0):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name in y_true_dict:
        y_bin = (y_true_dict[name] == pos_label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob_dict[name])
        auc = roc_auc_score(y_bin, y_prob_dict[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_metrics_bar(metrics_table, out_path):
    rows = []
    for model, splits in metrics_table.items():
        for split, mets in splits.items():
            for k, v in mets.items():
                rows.append({"model": model, "split": split, "metric": k, "value": v})
    df = pd.DataFrame(rows)
    keep = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for split in df["split"].unique():
        sub = df[(df["split"] == split) & (df["metric"].isin(keep))]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=sub, x="metric", y="value", hue="model", ax=ax)
        ax.set_ylim(0, 1.02); ax.set_title(f"Metrics on {split}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
        fig.tight_layout(); fig.savefig(out_path.parent / f"metrics_bar_{split}.png", dpi=150)
        plt.close(fig)
    df.to_csv(out_path, index=False)
