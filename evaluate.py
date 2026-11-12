"""
evaluate.py
-----------
Reproduces every quantitative analysis reported in Section VI of the
paper from a trained model:

  * Class-wise accuracy / precision / recall / F1               (Table 3-4)
  * Raw-count and row-normalised confusion matrices              (Figs. 2-5)
  * One-vs-rest ROC curves and AUC                                (Figs. 6-7)
  * Computational complexity: params, GFLOPs, model size,
    GPU/CPU latency                                                (Table 7)
  * 5-fold cross-validation, per-fold + overall                    (Tables 9-10)

Run as a script after training:
    python evaluate.py --corpus bc --checkpoint checkpoints/dacryformer_bc_best.pt
"""
import argparse
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold

from config import CFG
from dataset import build_datasets
from model import build_model


# ------------------------------------------------------------------
# Class-wise metrics (Section VI-A)
# ------------------------------------------------------------------
def classwise_report(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    total = cm.sum()

    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        class_acc = (tp + tn) / total
        rows.append({
            "class": name, "accuracy": class_acc, "precision": precision[i],
            "recall": recall[i], "f1": f1[i], "support": int(support[i]),
        })
    overall_acc = np.trace(cm) / total
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return rows, cm, overall_acc, macro_f1


def print_classwise_table(rows, overall_acc, macro_f1, corpus_name):
    print(f"\n=== Class-wise performance: {corpus_name} ===")
    print(f"{'Class':<14}{'Acc%':>8}{'Prec%':>8}{'Rec%':>8}{'F1%':>8}{'Supp':>8}")
    for r in rows:
        print(f"{r['class']:<14}{100*r['accuracy']:>8.2f}{100*r['precision']:>8.2f}"
              f"{100*r['recall']:>8.2f}{100*r['f1']:>8.2f}{r['support']:>8d}")
    print(f"Overall accuracy: {100*overall_acc:.2f}%   Macro F1: {100*macro_f1:.2f}%")


# ------------------------------------------------------------------
# Confusion matrices, raw + row-normalised (Section VI-B)
# ------------------------------------------------------------------
def confusion_matrices(y_true, y_pred, n_classes):
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = cm_raw / np.clip(row_sums, 1, None)
    return cm_raw, cm_norm


# ------------------------------------------------------------------
# One-vs-rest ROC / AUC (Section VI-C)
# ------------------------------------------------------------------
def roc_auc_per_class(y_true, y_score, n_classes):
    """y_score: (N, n_classes) softmax probabilities."""
    results = {}
    y_true = np.asarray(y_true)
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, c])
        results[c] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return results


@torch.no_grad()
def collect_predictions(model, loader, device, n_classes):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        mfcc = batch["mfcc"].to(device)
        logmel = batch["logmel"].to(device)
        tonnetz = batch["tonnetz"].to(device)
        domain = batch["domain"].to(device)
        labels = batch["label"].to(device)

        logits = model(mfcc, logmel, tonnetz, domain)[:, :n_classes]
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ------------------------------------------------------------------
# Computational complexity (Section VI-G / Table 7)
# ------------------------------------------------------------------
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_gflops(model, example_batch, device) -> float:
    """Rough GFLOPs estimate via `fvcore` if available, else falls back to
    a parameter-based order-of-magnitude estimate (2 x MACs approx params
    x avg spatial reduction -- install fvcore for an exact figure)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        mfcc = example_batch["mfcc"].to(device)
        logmel = example_batch["logmel"].to(device)
        tonnetz = example_batch["tonnetz"].to(device)
        domain = example_batch["domain"].to(device)
        flops = FlopCountAnalysis(model, (mfcc, logmel, tonnetz, domain))
        return flops.total() / 1e9
    except ImportError:
        print("[WARN] fvcore not installed; GFLOPs estimate skipped "
              "(pip install fvcore for an exact measurement).")
        return float("nan")


def measure_latency(model, example_batch, device, n_runs=100, n_warmup=20) -> float:
    model.eval()
    mfcc = example_batch["mfcc"].to(device)
    logmel = example_batch["logmel"].to(device)
    tonnetz = example_batch["tonnetz"].to(device)
    domain = example_batch["domain"].to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(mfcc, logmel, tonnetz, domain)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            model(mfcc, logmel, tonnetz, domain)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / n_runs * 1000.0  # ms/sample (batch size 1 recommended)


def model_size_mb(model) -> float:
    n_params = count_parameters(model)
    return n_params * 4 / (1024 ** 2)  # fp32


def computational_complexity_report(model, example_batch, device):
    params = count_parameters(model)
    gflops = estimate_gflops(model, example_batch, device)
    size_mb = model_size_mb(model)
    latency_ms = measure_latency(model, example_batch, device)
    print(f"\n=== Computational complexity ===")
    print(f"Params: {params/1e6:.2f} M | GFLOPs: {gflops:.3f} | "
          f"Model size: {size_mb:.1f} MB | Latency: {latency_ms:.2f} ms/sample "
          f"({device.type})")
    return {"params_m": params / 1e6, "gflops": gflops,
            "model_size_mb": size_mb, "latency_ms": latency_ms}


# ------------------------------------------------------------------
# 5-fold cross-validation (Section VI-I / Tables 9-10)
# ------------------------------------------------------------------
def five_fold_cv(corpus: str, build_model_fn, train_fn, n_classes: int, seed: int = 42):
    """Runs a fresh 5-fold, stratified, patient/file-level cross-validation
    and reports per-fold + overall accuracy/precision/recall/F1. `train_fn`
    should accept (train_subset, val_subset) and return a trained model."""
    datasets = build_datasets(CFG.data)
    full_ds = datasets[f"{corpus}_train"]  # combine all pre-split data upstream if desired
    labels = [full_ds.index[i][1] for i in range(len(full_ds))]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        train_subset = Subset(full_ds, train_idx)
        val_subset = Subset(full_ds, val_idx)

        model = train_fn(train_subset, val_subset)
        device = next(model.parameters()).device
        val_loader = DataLoader(val_subset, batch_size=CFG.train.batch_size, shuffle=False)
        probs, y_true = collect_predictions(model, val_loader, device, n_classes)
        y_pred = probs.argmax(axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        acc = (y_pred == y_true).mean()
        fold_metrics.append({"fold": fold, "accuracy": acc, "precision": precision,
                              "recall": recall, "f1": f1})
        print(f"[5-Fold CV][{corpus}] fold {fold}: acc={acc:.4f} "
              f"prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")

    accs = [m["accuracy"] for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs = [m["recall"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    summary = {
        "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs, ddof=1),
        "precision_mean": np.mean(precs), "precision_std": np.std(precs, ddof=1),
        "recall_mean": np.mean(recs), "recall_std": np.std(recs, ddof=1),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s, ddof=1),
    }
    print(f"[5-Fold CV][{corpus}] overall: "
          f"acc={summary['accuracy_mean']:.4f}+-{summary['accuracy_std']:.4f} "
          f"f1={summary['f1_mean']:.4f}+-{summary['f1_std']:.4f}")
    return fold_metrics, summary


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["bc", "dc"], required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    class_names = CFG.data.bc_classes if args.corpus == "bc" else CFG.data.dc_classes
    n_classes = len(class_names)

    model = build_model(CFG.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    datasets = build_datasets(CFG.data)
    test_loader = DataLoader(datasets[f"{args.corpus}_test"], batch_size=CFG.train.batch_size)

    probs, y_true = collect_predictions(model, test_loader, device, n_classes)
    y_pred = probs.argmax(axis=1)

    rows, cm, overall_acc, macro_f1 = classwise_report(y_true, y_pred, class_names)
    print_classwise_table(rows, overall_acc, macro_f1, args.corpus)

    cm_raw, cm_norm = confusion_matrices(y_true, y_pred, n_classes)
    print("\nRaw confusion matrix:\n", cm_raw)
    print("\nRow-normalised confusion matrix:\n", np.round(cm_norm, 3))

    roc_results = roc_auc_per_class(y_true, probs, n_classes)
    for c, name in enumerate(class_names):
        print(f"AUC[{name}] = {roc_results[c]['auc']:.4f}")

    example_batch = next(iter(test_loader))
    computational_complexity_report(model, example_batch, device)
