"""
losses.py
---------
Focal loss (Eq. 24), used to down-weight well-classified majority-class
examples given the class imbalance of both corpora (Section III). The L2
weight-decay term of Eq. 25 is applied via the AdamW optimizer's
`weight_decay` argument in train.py rather than added explicitly here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """L_focal = - sum_c alpha_c (1 - p_c)^gamma y_c log(p_c)   (Eq. 24)

    `class_weights` supplies alpha_c (inverse-frequency class weights,
    computed once per corpus from the training split -- see
    `compute_class_weights` below).
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        n_classes = logits.shape[-1]
        y_onehot = F.one_hot(targets, num_classes=n_classes).float()

        p_c = (probs * y_onehot).sum(dim=-1).clamp(min=1e-8)          # predicted prob of true class
        log_p_c = (log_probs * y_onehot).sum(dim=-1)
        alpha_c = self.alpha[targets]

        loss = -alpha_c * (1 - p_c).pow(self.gamma) * log_p_c
        return loss.mean()


def compute_class_weights(labels, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights alpha_c, normalised to sum to
    n_classes so the overall loss scale is comparable across corpora."""
    counts = torch.zeros(n_classes)
    for l in labels:
        counts[l] += 1
    counts = counts.clamp(min=1)
    inv_freq = 1.0 / counts
    alpha = inv_freq * (n_classes / inv_freq.sum())
    return alpha
