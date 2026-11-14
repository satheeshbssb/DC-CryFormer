from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from model import DACryFormer


# ------------------------------------------------------------------
# Forward-hook based activation/gradient capture
# ------------------------------------------------------------------
class ActivationGradHook:

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
    alpha = gradient.mean(dim=(2, 3), keepdim=True)        
    cam = F.relu((alpha * activation).sum(dim=1))           
    return cam


def _grad_cam_pp(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    grad2 = gradient.pow(2)
    grad3 = gradient.pow(3)
    sum_act = activation.sum(dim=(2, 3), keepdim=True)

    denom = 2 * grad2 + sum_act * grad3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom) * 1e-8)
    alpha_ij = grad2 / denom                                 

    weights = (F.relu(gradient) * alpha_ij).sum(dim=(2, 3))   
    cam = F.relu((weights.unsqueeze(-1).unsqueeze(-1) * activation).sum(dim=1))
    return cam


def _normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-8)
    return flat.view_as(cam)


@dataclass
class StreamCAMResult:
    gradcam: torch.Tensor     
    gradcam_pp: torch.Tensor   
    energy: float               


def compute_stream_cams(model: DACryFormer, batch, device, target_class=None) -> List[StreamCAMResult]:
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
# GGFG: periodic training-time feature gate
# ------------------------------------------------------------------
class GGFG:
    def __init__(self, update_interval: int = 5, lam: float = 0.5, n_probe_batches: int = 8):
        self.K = update_interval
        self.lam = lam
        self.n_probe_batches = n_probe_batches

    @torch.no_grad()
    def _write_gate(self, model: DACryFormer, energies: List[float]):
        eps = torch.tensor(energies, dtype=torch.float32)
        eps_norm = eps / (eps.sum() + 1e-8)               
        gate = 1.0 + self.lam * eps_norm                    
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
