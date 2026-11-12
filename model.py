"""
model.py
--------
DA-CryFormer architecture, Section IV of the paper.

    Stream encoders (A, Eqs. 4-7)
        -> Cross-Stream Tiny-Transformer fusion (B, Eqs. 8-10)
        -> BiGRU + attention pooling (C, Eqs. 11-15)
        -> Domain-Adaptive Normalisation (D, Eqs. 16-17)
        -> per-corpus Softmax head

The Grad-CAM/Grad-CAM++-Guided Feature Gate (E) is implemented separately
in gradcam.py, since it operates on gradients of the *already-built* graph
and is invoked periodically during training (every K epochs) rather than
being a static sub-module of the forward pass.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from config import ModelConfig


# ----------------------------------------------------------------------
# A. Depthwise-separable conv + SE stream encoder (Eqs. 4-7)
# ----------------------------------------------------------------------
class DSConvSEBlock(nn.Module):
    """One depthwise-separable convolution block with BatchNorm, Swish,
    and Squeeze-and-Excitation recalibration (Eqs. 4-7)."""

    def __init__(self, in_ch: int, out_ch: int, se_reduction: int = 8, stride: int = 2):
        super().__init__()
        # Eq. 4: depthwise conv (per-channel)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        # Eq. 5: pointwise (1x1) conv mixing channels
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()  # Swish

        # Eqs. 6-7: Squeeze-and-Excitation gate
        se_hidden = max(out_ch // se_reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(out_ch, se_hidden, kernel_size=1)
        self.se_fc2 = nn.Conv2d(se_hidden, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        y = self.act(x)                       # Y (post BN + Swish), Eq. 6 input

        # Eq. 6: z_c = (1/HW) sum_ij Y_{i,j,c}
        z = self.se_pool(y)
        # Eq. 7: s = sigma(W2 . delta(W1 . z)),  Y~_c = s_c . Y_{:,:,c}
        s = torch.sigmoid(self.se_fc2(Fnn.relu(self.se_fc1(z))))
        y_hat = s * y
        return y_hat


class StreamEncoder(nn.Module):
    """Stack of 4 DS-Conv+SE blocks (32->64->128->128 channels) followed by
    global average pooling and a Dense(256) projection -> stream embedding
    e_s in R^d (Section IV-A, final paragraph).

    Only the first two blocks downsample (stride=2); the last two use
    stride=1. This matters for the narrow CQT/Tonnetz stream (6 frequency
    bins): four consecutive stride-2 blocks would collapse it to a 1x1
    feature map, making Grad-CAM/Grad-CAM++ spatially degenerate for that
    stream (see gradcam.py). Two stride-2 blocks keep at least a 2x1 map
    for a 6-bin input while still giving MFCC (39 bins) and log-Mel
    (40 bins) a meaningful spatial reduction.
    """

    def __init__(self, cfg: ModelConfig, in_channels: int = 1):
        super().__init__()
        chans = [in_channels] + list(cfg.stream_channels)
        strides = [2, 2] + [1] * max(0, len(cfg.stream_channels) - 2)
        blocks = []
        for i in range(len(cfg.stream_channels)):
            blocks.append(DSConvSEBlock(chans[i], chans[i + 1],
                                         se_reduction=cfg.se_reduction,
                                         stride=strides[i]))
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(cfg.stream_channels[-1], cfg.embed_dim)

        self._last_feature_map = None  # cached for Grad-CAM (see gradcam.py)

    def forward(self, x, return_feature_map: bool = False):
        for blk in self.blocks:
            x = blk(x)
        self._last_feature_map = x  # (B, C, H', W') -- Grad-CAM target layer
        pooled = self.gap(x).flatten(1)
        embed = self.proj(pooled)
        if return_feature_map:
            return embed, x
        return embed


# ----------------------------------------------------------------------
# B. Cross-Stream Tiny-Transformer fusion (Eqs. 8-10)
# ----------------------------------------------------------------------
class CrossStreamTinyTransformer(nn.Module):
    """Treats the 3 stream embeddings as a length-3 token sequence and lets
    them attend to one another via a compact pre-norm Transformer encoder
    (Eqs. 8-10), then mean-pools the 3 output tokens into the fused vector f."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.cstf_heads,
            dim_feedforward=cfg.cstf_ffn_dim,
            dropout=cfg.cstf_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm, as stated in the paper
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.cstf_layers)

    def forward(self, e1, e2, e3):
        # E = [e_1; e_2; e_3] in R^{3 x d}, batched as (B, 3, d)
        E = torch.stack([e1, e2, e3], dim=1)
        out = self.encoder(E)                 # Eqs. 8-9 (multi-head self-attn)
        f = out.mean(dim=1)                   # Eq. 10: mean-pool 3 tokens
        return f, out                         # also return per-stream tokens for GGFG


# ----------------------------------------------------------------------
# C. BiGRU temporal encoder + attention pooling (Eqs. 11-15)
# ----------------------------------------------------------------------
class BiGRUAttentionPool(nn.Module):
    """Bidirectional GRU (Eqs. 11-14) over the T fused per-window
    embeddings, followed by additive attention pooling (Eq. 15)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = cfg.gru_hidden * 2
        self.attn_proj = nn.Linear(out_dim, out_dim)   # W_a, b_a
        self.attn_context = nn.Linear(out_dim, 1, bias=False)  # w_c

    def forward(self, f_seq):
        # f_seq: (B, T, d) sequence of fused per-window embeddings
        h, _ = self.gru(f_seq)                          # Eqs. 11-14 -> h_t = [->h_t ; <-h_t]
        g = torch.tanh(self.attn_proj(h))                # Eq. 15
        alpha_logits = self.attn_context(g).squeeze(-1)  # (B, T)
        alpha = torch.softmax(alpha_logits, dim=1)
        v = torch.einsum("bt,btd->bd", alpha, h)         # v = sum_t alpha_t h_t
        return v, alpha


# ----------------------------------------------------------------------
# D. Domain-Adaptive Normalisation (Eqs. 16-17)
# ----------------------------------------------------------------------
class DomainAdaptiveNorm(nn.Module):
    """FiLM-style conditioning: standardises v, then applies a per-domain
    scale/shift predicted from a learned domain embedding (Eqs. 16-17)."""

    def __init__(self, cfg: ModelConfig, feat_dim: int):
        super().__init__()
        self.domain_embed = nn.Embedding(cfg.n_domains, cfg.domain_embed_dim)
        self.to_gamma = nn.Linear(cfg.domain_embed_dim, feat_dim)
        self.to_beta = nn.Linear(cfg.domain_embed_dim, feat_dim)
        self.eps = 1e-5

    def forward(self, v, domain_ids):
        mu = v.mean(dim=-1, keepdim=True)
        var = v.var(dim=-1, unbiased=False, keepdim=True)
        v_hat = (v - mu) / torch.sqrt(var + self.eps)     # Eq. 16

        d = self.domain_embed(domain_ids)                  # (B, domain_embed_dim)
        gamma = self.to_gamma(d)
        beta = self.to_beta(d)
        v_prime = gamma * v_hat + beta                      # Eq. 17
        return v_prime


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class DACryFormer(nn.Module):
    """End-to-end DA-CryFormer. Streams are encoded per analysis window
    (T windows / 1-s segment), fused per window by the CSTF, aggregated
    over time by the BiGRU + attention pool, domain-normalised, and
    classified by one of two per-corpus Softmax heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_mfcc = StreamEncoder(cfg, in_channels=1)
        self.enc_logmel = StreamEncoder(cfg, in_channels=1)
        self.enc_tonnetz = StreamEncoder(cfg, in_channels=1)

        self.cstf = CrossStreamTinyTransformer(cfg)
        self.temporal = BiGRUAttentionPool(cfg)

        feat_dim = cfg.gru_hidden * 2
        self.dan = DomainAdaptiveNorm(cfg, feat_dim)

        self.dropout = nn.Dropout(cfg.classifier_dropout)
        self.head_bc = nn.Linear(feat_dim, cfg.n_classes_bc)
        self.head_dc = nn.Linear(feat_dim, cfg.n_classes_dc)

        # Learnable per-stream gate applied by the GGFG module (Eq. 23);
        # kept as a buffer of ones by default (no-op) and updated by
        # gradcam.py every K epochs during training.
        self.register_buffer("stream_gate", torch.ones(3))

    def encode_streams(self, mfcc, logmel, tonnetz, return_feature_maps=False):
        """mfcc/logmel/tonnetz: (B, 1, F, T_windows) time-frequency maps
        for ONE analysis window. In practice this is called T times (once
        per window) by `forward`, or vectorised by folding T into the
        batch dimension -- see `forward` below for the vectorised path."""
        if return_feature_maps:
            e1, fmap1 = self.enc_mfcc(mfcc, return_feature_map=True)
            e2, fmap2 = self.enc_logmel(logmel, return_feature_map=True)
            e3, fmap3 = self.enc_tonnetz(tonnetz, return_feature_map=True)
            return (e1, e2, e3), (fmap1, fmap2, fmap3)
        e1 = self.enc_mfcc(mfcc)
        e2 = self.enc_logmel(logmel)
        e3 = self.enc_tonnetz(tonnetz)
        return e1, e2, e3

    def apply_stream_gate(self, e1, e2, e3):
        """Eq. 23: e'_s = (1 + lambda * eps~_s) . e_s. `stream_gate`
        already stores (1 + lambda * eps~_s) per stream, refreshed by
        gradcam.GGFG.update() every K epochs."""
        g = self.stream_gate
        return e1 * g[0], e2 * g[1], e3 * g[2]

    def forward(self, mfcc, logmel, tonnetz, domain_ids, return_attn=False):
        """
        mfcc:    (B, T, 1, F_mfcc, win)   -- T = cfg.n_windows analysis windows
        logmel:  (B, T, 1, F_mel, win)
        tonnetz: (B, T, 1, F_tonnetz, win)
        domain_ids: (B,) long tensor, 0 = BC, 1 = DC

        Each of the T windows is encoded independently by the (weight-shared
        across windows) stream encoders, fused per window by the CSTF, and
        the resulting T fused embeddings are aggregated by the BiGRU +
        attention pool (Section IV-C).
        """
        B, T = mfcc.shape[0], mfcc.shape[1]

        def fold(x):
            # (B, T, 1, F, win) -> (B*T, 1, F, win)
            return x.reshape(B * T, *x.shape[2:])

        mfcc_f, logmel_f, tonnetz_f = fold(mfcc), fold(logmel), fold(tonnetz)

        e1, e2, e3 = self.encode_streams(mfcc_f, logmel_f, tonnetz_f)   # (B*T, d) each
        e1, e2, e3 = self.apply_stream_gate(e1, e2, e3)

        f, _ = self.cstf(e1, e2, e3)              # (B*T, d) fused per-window embedding
        f_seq = f.view(B, T, -1)                   # (B, T, d)

        v, alpha = self.temporal(f_seq)             # Eqs. 11-15 -> (B, feat_dim)
        v_prime = self.dan(v, domain_ids)            # Eqs. 16-17
        v_prime = self.dropout(v_prime)

        logits_bc = self.head_bc(v_prime)
        logits_dc = self.head_dc(v_prime)
        # Select the head matching each sample's domain.
        logits = torch.where(
            (domain_ids == 0).unsqueeze(-1),
            _pad_like(logits_bc, logits_dc),
            _pad_like(logits_dc, logits_bc),
        )
        if return_attn:
            return logits, alpha
        return logits


def _pad_like(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Right-pads `a`'s last dim with -inf to match `b`'s width, so the two
    per-corpus heads (possibly with different class counts) can share one
    `torch.where` selection without a shape mismatch."""
    if a.shape[-1] == b.shape[-1]:
        return a
    pad = b.shape[-1] - a.shape[-1]
    return Fnn.pad(a, (0, pad), value=float("-inf"))


def build_model(cfg: ModelConfig) -> DACryFormer:
    model = DACryFormer(cfg)
    return model
