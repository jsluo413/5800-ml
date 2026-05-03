import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.dataset import find_split_dirs, make_loaders
from src.evaluate import (collect_predictions, compute_metrics,
                          plot_confusion_matrix, plot_metrics_bar, plot_roc,
                          plot_training_curves)
from src.gradcam import make_gradcam_grid
from src.models import build_model, count_params
from src.train import train_model

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "datafolder"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
CKPTS = RESULTS / "checkpoints"
METRICS_DIR = RESULTS / "metrics"
GRADCAM_DIR = RESULTS / "gradcam"

MODEL_NAMES = ["simplecnn", "resnet18", "efficientnet_b0", "mobilenet_v3_small"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_gradcam_examples(model, loader, device, pos_label, n_per_group=4):
    _, preds, _ = collect_predictions(model, loader, device, pos_label=pos_label)
    idx_to_class = {v: k for k, v in loader.dataset.class_to_idx.items()}
    paths = [Path(s[0]) for s in loader.dataset.samples]
    targets = np.array([s[1] for s in loader.dataset.samples])

    def take(mask, tag):
        idxs = np.where(mask)[0][:n_per_group]
        return [(paths[i], idx_to_class[int(targets[i])], tag) for i in idxs]

    correct = preds == targets
    return (take(correct & (targets == pos_label), "Correct: damage")
            + take(correct & (targets != pos_label), "Correct: no_damage")
            + take((~correct) & (targets == pos_label), "FN: missed damage")
            + take((~correct) & (targets != pos_label), "FP: false alarm"))


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, test_bal, test_unbal = make_loaders(
        DATA_ROOT, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"sizes  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test_bal={len(test_bal.dataset)}  test_unbal={len(test_unbal.dataset)}")

    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pos_label = class_to_idx["damage"]
    print("class_to_idx:", class_to_idx)

    histories, predictions, metrics_table, model_objs, target_layers = {}, {}, {}, {}, {}

    for name in MODEL_NAMES:
        print(f"\n[{name}]")
        model, target_layer = build_model(name)
        model.to(device)
        n_params = count_params(model)
        print("  params:", f"{n_params:,}")

        epochs = args.epochs_baseline if name == "simplecnn" else args.epochs_pretrained
        lr = args.lr_baseline if name == "simplecnn" else args.lr_pretrained
        hist = train_model(model, name, train_loader, val_loader, device, CKPTS,
                           epochs=epochs, lr=lr)
        histories[name] = hist
        model_objs[name] = model
        target_layers[name] = target_layer

        per_split_metrics, per_split_preds = {}, {}
        for split_name, loader in [("test_balanced", test_bal),
                                   ("test_unbalanced", test_unbal)]:
            probs, preds, targets = collect_predictions(model, loader, device, pos_label=pos_label)
            mets = compute_metrics(targets, preds, probs, pos_label=pos_label)
            mets["n"] = int(len(targets))
            mets["params"] = int(n_params)
            per_split_metrics[split_name] = mets
            per_split_preds[split_name] = (probs, preds, targets)
            print(f"  {split_name}:",
                  " ".join(f"{k}={v:.4f}" for k, v in mets.items() if isinstance(v, float)))
            cm = confusion_matrix(targets, preds)
            order = [idx_to_class[i] for i in range(len(idx_to_class))]
            plot_confusion_matrix(cm, order, f"{name} | {split_name}",
                                  FIGS / f"cm_{name}_{split_name}.png")
        metrics_table[name] = per_split_metrics
        predictions[name] = per_split_preds

    plot_training_curves(histories, FIGS / "training_curves.png")
    plot_metrics_bar(metrics_table, METRICS_DIR / "metrics_summary.csv")

    for split_name in ["test_balanced", "test_unbalanced"]:
        y_true_d = {n: predictions[n][split_name][2] for n in MODEL_NAMES}
        y_prob_d = {n: predictions[n][split_name][0] for n in MODEL_NAMES}
        plot_roc(y_true_d, y_prob_d, f"ROC | {split_name}",
                 FIGS / f"roc_{split_name}.png", pos_label=pos_label)

    with open(METRICS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_table, f, indent=2)

    best_name = max(metrics_table, key=lambda n: metrics_table[n]["test_balanced"]["f1"])
    print("\nbest model (F1 on balanced test):", best_name)
    examples = pick_gradcam_examples(model_objs[best_name], test_bal, device,
                                     pos_label=pos_label, n_per_group=4)
    make_gradcam_grid(model_objs[best_name], target_layers[best_name], device,
                      examples, [idx_to_class[i] for i in range(2)],
                      GRADCAM_DIR / f"gradcam_best_{best_name}.png",
                      title=f"Grad-CAM | {best_name} (best)")

    sample_imgs = examples[:8]
    for name in MODEL_NAMES:
        make_gradcam_grid(model_objs[name], target_layers[name], device,
                          sample_imgs, [idx_to_class[i] for i in range(2)],
                          GRADCAM_DIR / f"gradcam_{name}.png",
                          title=f"Grad-CAM | {name}")

    rows = []
    for name, splits_d in metrics_table.items():
        for split, mets in splits_d.items():
            rows.append({"model": name, "split": split, **mets})
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "summary_table.csv", index=False)
    print("\nsummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs_baseline", type=int, default=8)
    p.add_argument("--epochs_pretrained", type=int, default=5)
    p.add_argument("--lr_baseline", type=float, default=1e-3)
    p.add_argument("--lr_pretrained", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    main(p.parse_args())
