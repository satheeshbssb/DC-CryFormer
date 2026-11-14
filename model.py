import math
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from config import ModelConfig


# ----------------------------------------------------------------------
# A. Depthwise-separable conv + SE stream encoder
# ----------------------------------------------------------------------
class DSConvSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, se_reduction: int = 8, stride: int = 2):
        super().__init__()
        # depthwise conv (per-channel)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        # pointwise (1x1) conv mixing channels
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()  # Swish

        # Squeeze-and-Excitation gate
        se_hidden = max(out_ch // se_reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(out_ch, se_hidden, kernel_size=1)
        self.se_fc2 = nn.Conv2d(se_hidden, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        y = self.act(x)                       

        # Eq. 6: z_c = (1/HW) sum_ij Y_{i,j,c}
        z = self.se_pool(y)
        # Eq. 7: s = sigma(W2 . delta(W1 . z)),  Y~_c = s_c . Y_{:,:,c}
        s = torch.sigmoid(self.se_fc2(Fnn.relu(self.se_fc1(z))))
        y_hat = s * y
        return y_hat


class StreamEncoder(nn.Module):
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

        self._last_feature_map = None  

    def forward(self, x, return_feature_map: bool = False):
        for blk in self.blocks:
            x = blk(x)
        self._last_feature_map = x  
        pooled = self.gap(x).flatten(1)
        embed = self.proj(pooled)
        if return_feature_map:
            return embed, x
        return embed


# ----------------------------------------------------------------------
# B. Cross-Stream Tiny-Transformer fusion (Eqs. 8-10)
# ----------------------------------------------------------------------
class CrossStreamTinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.cstf_heads,
            dim_feedforward=cfg.cstf_ffn_dim,
            dropout=cfg.cstf_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.cstf_layers)

    def forward(self, e1, e2, e3):
        
        E = torch.stack([e1, e2, e3], dim=1)
        out = self.encoder(E)               
        f = out.mean(dim=1)                
        return f, out                       


# ----------------------------------------------------------------------
# C. BiGRU temporal encoder + attention pooling
# ----------------------------------------------------------------------
class BiGRUAttentionPool(nn.Module):
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
        self.attn_proj = nn.Linear(out_dim, out_dim)   
        self.attn_context = nn.Linear(out_dim, 1, bias=False)  

    def forward(self, f_seq):
        h, _ = self.gru(f_seq)                          
        g = torch.tanh(self.attn_proj(h))              
        alpha_logits = self.attn_context(g).squeeze(-1) 
        alpha = torch.softmax(alpha_logits, dim=1)
        v = torch.einsum("bt,btd->bd", alpha, h)      
        return v, alpha


# ----------------------------------------------------------------------
# D. Domain-Adaptive Normalisation (Eqs. 16-17)
# ----------------------------------------------------------------------
class DomainAdaptiveNorm(nn.Module):
    def __init__(self, cfg: ModelConfig, feat_dim: int):
        super().__init__()
        self.domain_embed = nn.Embedding(cfg.n_domains, cfg.domain_embed_dim)
        self.to_gamma = nn.Linear(cfg.domain_embed_dim, feat_dim)
        self.to_beta = nn.Linear(cfg.domain_embed_dim, feat_dim)
        self.eps = 1e-5

    def forward(self, v, domain_ids):
        mu = v.mean(dim=-1, keepdim=True)
        var = v.var(dim=-1, unbiased=False, keepdim=True)
        v_hat = (v - mu) / torch.sqrt(var + self.eps)    

        d = self.domain_embed(domain_ids)                
        gamma = self.to_gamma(d)
        beta = self.to_beta(d)
        v_prime = gamma * v_hat + beta                 
        return v_prime


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class DACryFormer(nn.Module):
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
        self.register_buffer("stream_gate", torch.ones(3))

    def encode_streams(self, mfcc, logmel, tonnetz, return_feature_maps=False):
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
        g = self.stream_gate
        return e1 * g[0], e2 * g[1], e3 * g[2]

    def forward(self, mfcc, logmel, tonnetz, domain_ids, return_attn=False):
         B, T = mfcc.shape[0], mfcc.shape[1]
        def fold(x):           
            return x.reshape(B * T, *x.shape[2:])
        mfcc_f, logmel_f, tonnetz_f = fold(mfcc), fold(logmel), fold(tonnetz)
        e1, e2, e3 = self.encode_streams(mfcc_f, logmel_f, tonnetz_f)   
        e1, e2, e3 = self.apply_stream_gate(e1, e2, e3)

        f, _ = self.cstf(e1, e2, e3)         
        f_seq = f.view(B, T, -1)             

        v, alpha = self.temporal(f_seq)          
        v_prime = self.dan(v, domain_ids)        
        v_prime = self.dropout(v_prime)

        logits_bc = self.head_bc(v_prime)
        logits_dc = self.head_dc(v_prime)
       
        logits = torch.where(
            (domain_ids == 0).unsqueeze(-1),
            _pad_like(logits_bc, logits_dc),
            _pad_like(logits_dc, logits_bc),
        )
        if return_attn:
            return logits, alpha
        return logits


def _pad_like(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[-1] == b.shape[-1]:
        return a
    pad = b.shape[-1] - a.shape[-1]
    return Fnn.pad(a, (0, pad), value=float("-inf"))


def build_model(cfg: ModelConfig) -> DACryFormer:
    model = DACryFormer(cfg)
    return model
