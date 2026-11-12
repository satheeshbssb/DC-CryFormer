"""
gradcam.py
----------
Implements:
  * Grad-CAM (Eqs. 18-19)          -- Selvaraju et al., 2017
  * Grad-CAM++ (Eqs. 20-21)        -- Chattopadhay et al., 2018
  * The Grad-CAM/Grad-CAM++-Guided Feature Gate, GGFG (Eqs. 22-23,
    Algorithm 1 lines 17-21), which recomputes stream-level activation
    energy every K epochs on a held-out validation subset and updates
    `model.stream_gate` accordingly.

The Grad-CAM++ weights use the standard closed-form approximation adopted
by widely-used libraries such as pytorch-grad-cam (also used for the
qualitative visualisations in the paper): the true 2nd/3rd-order partial
derivatives of Eq. 20 are approximated algebraically from the first-order
gradient, which is exact under a global-average-pool + linear read-out
assumption and is the standard practical implementation of Grad-CAM++.
"""
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from model import DACryFormer


# ------------------------------------------------------------------
# Forward-hook based activation/gradient capture
# ------------------------------------------------------------------
class ActivationGradHook:
    """Captures the forward activation of a target conv layer and its
    gradient during backward (retains grad on a non-leaf tensor)."""

    def __init__(self, module: torch.nn.Module):
        self.activation = None
        self.gradient = None
        self._fwd_handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activation = out
        out.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        self.gradient = grad

    def remove(self):
        self._fwd_handle.remove()


def _grad_cam(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Eq. 18-19. activation, gradient: (N, C, H, W) -> returns (N, H, W)."""
    alpha = gradient.mean(dim=(2, 3), keepdim=True)         # Eq. 18
    cam = F.relu((alpha * activation).sum(dim=1))            # Eq. 19
    return cam


def _grad_cam_pp(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Eq. 20-21, standard closed-form implementation. Returns (N, H, W)."""
    grad2 = gradient.pow(2)
    grad3 = gradient.pow(3)
    sum_act = activation.sum(dim=(2, 3), keepdim=True)

    denom = 2 * grad2 + sum_act * grad3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom) * 1e-8)
    alpha_ij = grad2 / denom                                  # Eq. 20 (closed-form)

    weights = (F.relu(gradient) * alpha_ij).sum(dim=(2, 3))   # Eq. 21, w_k^c
    cam = F.relu((weights.unsqueeze(-1).unsqueeze(-1) * activation).sum(dim=1))
    return cam


def _normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalisation to [0, 1]."""
    flat = cam.flatten(1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-8)
    return flat.view_as(cam)


@dataclass
class StreamCAMResult:
    gradcam: torch.Tensor      # (N, H, W), normalised
    gradcam_pp: torch.Tensor   # (N, H, W), normalised
    energy: float               # mean of 0.5*(gradcam) + 0.5*(gradcam++), Eq. 22 numerator


def compute_stream_cams(model: DACryFormer, batch, device, target_class=None) -> List[StreamCAMResult]:
    """Runs one forward/backward pass and returns Grad-CAM / Grad-CAM++
    maps for all three streams (MFCC, log-Mel, CQT/Tonnetz), targeting the
    predicted class by default (or `target_class` if given)."""
    model.eval()

    hooks = [
        ActivationGradHook(model.enc_mfcc.blocks[-1]),
        ActivationGradHook(model.enc_logmel.blocks[-1]),
        ActivationGradHook(model.enc_tonnetz.blocks[-1]),
    ]

    mfcc = batch["mfcc"].to(device).requires_grad_(False)
    logmel = batch["logmel"].to(device)
    tonnetz = batch["tonnetz"].to(device)
    domain = batch["domain"].to(device)

    logits = model(mfcc, logmel, tonnetz, domain)
    if target_class is None:
        target_class = logits.argmax(dim=-1)
    else:
        target_class = torch.full((logits.shape[0],), target_class,
                                   dtype=torch.long, device=device)

    scores = logits.gather(1, target_class.unsqueeze(1)).sum()
    model.zero_grad(set_to_none=True)
    scores.backward()

    results = []
    for h in hooks:
        act, grad = h.activation.detach(), h.gradient.detach()
        gc = _normalise_cam(_grad_cam(act, grad))
        gcpp = _normalise_cam(_grad_cam_pp(act, grad))
        energy = float((0.5 * gc.mean() + 0.5 * gcpp.mean()).item())
        results.append(StreamCAMResult(gradcam=gc.cpu(), gradcam_pp=gcpp.cpu(), energy=energy))
        h.remove()

    return results


# ------------------------------------------------------------------
# GGFG: periodic training-time feature gate (Eqs. 22-23)
# ------------------------------------------------------------------
class GGFG:
    """Grad-CAM/Grad-CAM++-Guided Feature Gate. Call `.maybe_update(...)`
    once per epoch; it recomputes the stream-recalibration gate every K
    epochs (Section IV-E, Algorithm 1 lines 17-21) on a fixed held-out
    validation subset and writes the result into `model.stream_gate`."""

    def __init__(self, update_interval: int = 5, lam: float = 0.5, n_probe_batches: int = 8):
        self.K = update_interval
        self.lam = lam
        self.n_probe_batches = n_probe_batches

    @torch.no_grad()
    def _write_gate(self, model: DACryFormer, energies: List[float]):
        eps = torch.tensor(energies, dtype=torch.float32)
        eps_norm = eps / (eps.sum() + 1e-8)                 # eps~_s, Eq. 22
        gate = 1.0 + self.lam * eps_norm                     # Eq. 23
        model.stream_gate.copy_(gate.to(model.stream_gate.device))

    def maybe_update(self, model: DACryFormer, val_loader, device, epoch: int):
        if epoch == 0 or epoch % self.K != 0:
            return None
        sum_energy = [0.0, 0.0, 0.0]
        n = 0
        for i, batch in enumerate(val_loader):
            if i >= self.n_probe_batches:
                break
            cams = compute_stream_cams(model, batch, device)
            for s in range(3):
                sum_energy[s] += cams[s].energy
            n += 1
        if n == 0:
            return None
        mean_energy = [e / n for e in sum_energy]
        self._write_gate(model, mean_energy)
        model.train()
        return mean_energy
