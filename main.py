import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import build_datasets
from train import train_one_corpus
from evaluate import (classwise_report, print_classwise_table, confusion_matrices,
                       roc_auc_per_class, collect_predictions, computational_complexity_report)
from plotting import plot_confusion_raw, plot_confusion_norm, plot_roc_curves

OUT_DIR = "./outputs"


def run_corpus(corpus: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    class_names = list(CFG.data.bc_classes if corpus == "bc" else CFG.data.dc_classes)
    n_classes = len(class_names)

    model, test_metrics = train_one_corpus(corpus)
    device = next(model.parameters()).device

    datasets = build_datasets(CFG.data)
    test_loader = DataLoader(datasets[f"{corpus}_test"], batch_size=CFG.train.batch_size)

    probs, y_true = collect_predictions(model, test_loader, device, n_classes)
    y_pred = probs.argmax(axis=1)

    rows, cm, overall_acc, macro_f1 = classwise_report(y_true, y_pred, class_names)
    print_classwise_table(rows, overall_acc, macro_f1, corpus)

    cm_raw, cm_norm = confusion_matrices(y_true, y_pred, n_classes)
    plot_confusion_raw(cm_raw, class_names, f"Confusion Matrix (Raw) - {corpus.upper()}",
                        os.path.join(OUT_DIR, f"cm_raw_{corpus}.png"))
    plot_confusion_norm(cm_norm, class_names, f"Confusion Matrix (Normalised) - {corpus.upper()}",
                         os.path.join(OUT_DIR, f"cm_norm_{corpus}.png"))

    roc_results = roc_auc_per_class(y_true, probs, n_classes)
    plot_roc_curves(roc_results, class_names, f"ROC Curves - {corpus.upper()}",
                     os.path.join(OUT_DIR, f"roc_{corpus}.png"))

    example_batch = next(iter(test_loader))
    computational_complexity_report(model, example_batch, device)

    torch.save(model.state_dict(), os.path.join(CFG.train.checkpoint_dir, f"dacryformer_{corpus}_final.pt"))
    print(f"[DONE] corpus={corpus}: test_acc={test_metrics['accuracy']:.4f} "
          f"test_macroF1={test_metrics['macro_f1']:.4f}. Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["bc", "dc", "both"], default="both")
    args = parser.parse_args()

    if args.corpus in ("bc", "both"):
        run_corpus("bc")
    if args.corpus in ("dc", "both"):
        run_corpus("dc")
