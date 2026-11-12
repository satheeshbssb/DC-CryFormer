"""
train.py
--------
End-to-end training loop for DA-CryFormer, implementing Algorithm 1 of the
paper: per-epoch iteration over mini-batches, forward pass through the
three stream encoders -> CSTF -> BiGRU+attention -> DAN -> per-corpus head,
periodic GGFG stream-gate recalibration (every K epochs), focal-loss +
weight-decay optimisation with AdamW, cosine-annealed LR with warm-up, and
early stopping on validation macro-F1 (TABLE 5 / Section IV-G).

Usage:
    python train.py --corpus bc      # train & evaluate on Baby Chillanto
    python train.py --corpus dc      # train & evaluate on Donate-a-Cry
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config import CFG
from dataset import build_datasets, DOMAIN_BC, DOMAIN_DC
from model import build_model
from losses import FocalLoss, compute_class_weights
from gradcam import GGFG


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_n = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in loader:
        mfcc = batch["mfcc"].to(device)
        logmel = batch["logmel"].to(device)
        tonnetz = batch["tonnetz"].to(device)
        domain = batch["domain"].to(device)
        labels = batch["label"].to(device)

        logits = model(mfcc, logmel, tonnetz, domain)
        logits = logits[:, :n_classes]
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"loss": total_loss / max(1, total_n), "accuracy": acc, "macro_f1": macro_f1,
            "preds": preds, "labels": labels}


def train_one_corpus(corpus: str):
    assert corpus in ("bc", "dc")
    set_seed(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on corpus={corpus}, device={device}")

    # ---- Data --------------------------------------------------------
    datasets = build_datasets(CFG.data)
    train_ds = datasets[f"{corpus}_train"]
    val_ds = datasets[f"{corpus}_val"]
    test_ds = datasets[f"{corpus}_test"]

    train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True,
                               num_workers=CFG.train.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.train.batch_size, shuffle=False,
                             num_workers=CFG.train.num_workers)
    test_loader = DataLoader(test_ds, batch_size=CFG.train.batch_size, shuffle=False,
                              num_workers=CFG.train.num_workers)

    n_classes = CFG.model.n_classes_bc if corpus == "bc" else CFG.model.n_classes_dc

    # ---- Model, loss, optimiser ---------------------------------------
    model = build_model(CFG.model).to(device)

    train_labels = [train_ds.index[i][1] for i in range(len(train_ds))]
    class_weights = compute_class_weights(train_labels, n_classes).to(device)
    criterion = FocalLoss(class_weights, gamma=CFG.train.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.train.lr,
                                   weight_decay=CFG.train.weight_decay)
    scheduler = get_lr_scheduler(optimizer, CFG.train.warmup_epochs,
                                  CFG.train.epochs, len(train_loader))

    ggfg = GGFG(update_interval=CFG.model.ggfg_update_interval,
                lam=CFG.model.ggfg_lambda)

    # ---- Training loop (Algorithm 1) -----------------------------------
    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    os.makedirs(CFG.train.checkpoint_dir, exist_ok=True)

    for epoch in range(1, CFG.train.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, n_seen = 0.0, 0

        # Algorithm 1, lines 17-21: periodic Grad-CAM/Grad-CAM++ gate update
        energies = ggfg.maybe_update(model, val_loader, device, epoch)
        if energies is not None:
            print(f"[GGFG] epoch {epoch}: stream energies (MFCC, logMel, CQT) = "
                  f"{[round(e, 4) for e in energies]}")

        for batch in train_loader:
            mfcc = batch["mfcc"].to(device)
            logmel = batch["logmel"].to(device)
            tonnetz = batch["tonnetz"].to(device)
            domain = batch["domain"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mfcc, logmel, tonnetz, domain)
            logits = logits[:, :n_classes]

            loss = criterion(logits, labels)               # Eq. 24 (+ Eq. 25 via weight_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        val_metrics = evaluate(model, val_loader, device, n_classes)
        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}/{CFG.train.epochs}] "
              f"train_loss={running_loss / max(1, n_seen):.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_macroF1={val_metrics['macro_f1']:.4f} "
              f"({dt:.1f}s)")

        # ---- Early stopping on validation macro-F1 ------------------------
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state, os.path.join(CFG.train.checkpoint_dir, f"dacryformer_{corpus}_best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CFG.train.early_stopping_patience:
                print(f"[INFO] Early stopping at epoch {epoch} "
                      f"(no val macro-F1 improvement for {CFG.train.early_stopping_patience} epochs).")
                break

    # ---- Final test-set evaluation with best checkpoint --------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, n_classes)
    print(f"[TEST] corpus={corpus} accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["bc", "dc"], required=True,
                         help="bc = Baby Chillanto, dc = Donate-a-Cry")
    args = parser.parse_args()
    train_one_corpus(args.corpus)
