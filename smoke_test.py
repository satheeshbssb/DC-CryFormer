"""
smoke_test.py
-------------
Sanity-checks the model architecture, loss, and GGFG gate with synthetic
tensors of the correct shapes (no real audio / corpus required). Not part
of the paper pipeline -- purely a developer utility to catch shape bugs.
"""
import torch
from config import CFG
from model import build_model
from losses import FocalLoss, compute_class_weights
from gradcam import compute_stream_cams, GGFG

torch.manual_seed(0)


def fake_batch(batch_size, cfg):
    T = cfg.data.n_windows
    win = 8  # arbitrary small window length for the smoke test
    mfcc = torch.randn(batch_size, T, 1, 39, win)
    logmel = torch.randn(batch_size, T, 1, cfg.data.n_mels, win)
    tonnetz = torch.randn(batch_size, T, 1, 6, win)
    labels = torch.randint(0, 5, (batch_size,))
    domain = torch.randint(0, 2, (batch_size,))
    return {"mfcc": mfcc, "logmel": logmel, "tonnetz": tonnetz,
            "label": labels, "domain": domain}


def main():
    device = torch.device("cpu")
    model = build_model(CFG.model).to(device)
    print(f"[OK] model built, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    batch = fake_batch(4, CFG)
    logits = model(batch["mfcc"], batch["logmel"], batch["tonnetz"], batch["domain"])
    print(f"[OK] forward pass -> logits shape {tuple(logits.shape)}")
    assert logits.shape == (4, max(CFG.model.n_classes_bc, CFG.model.n_classes_dc))

    weights = compute_class_weights(batch["label"].tolist(), CFG.model.n_classes_bc)
    criterion = FocalLoss(weights, gamma=CFG.train.focal_gamma)
    loss = criterion(logits[:, :CFG.model.n_classes_bc], batch["label"])
    loss.backward()
    print(f"[OK] backward pass -> loss={loss.item():.4f}, grads populated: "
          f"{model.enc_mfcc.blocks[0].pointwise.weight.grad is not None}")

    cams = compute_stream_cams(model, batch, device)
    for name, res in zip(["MFCC", "logMel", "CQT/Tonnetz"], cams):
        print(f"[OK] Grad-CAM[{name}]: shape={tuple(res.gradcam.shape)}, "
              f"Grad-CAM++ shape={tuple(res.gradcam_pp.shape)}, energy={res.energy:.4f}")

    ggfg = GGFG(update_interval=1, lam=0.5, n_probe_batches=1)
    loader = [batch]  # a single fake "val loader" batch
    energies = ggfg.maybe_update(model, loader, device, epoch=1)
    print(f"[OK] GGFG update -> stream_gate={model.stream_gate.tolist()}, energies={energies}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
