"""
config.py
---------
Central configuration for DA-CryFormer, matching TABLE 5 ("Hyperparameters
used for training DA-CryFormer") and Section IV / V of the paper:

    "DA-CryFormer: A Domain-Adaptive, Grad-CAM/Grad-CAM++-Guided
    Cross-Corpus Transformer-BiGRU Network for Infant Cry Classification
    Across Clinical and Crowd-Sourced Acoustic Datasets"

All numeric values below are taken directly from the paper. Edit paths in
`DataConfig` to point at your local copies of the Baby Chillanto (BC) and
Donate-a-Cry (DC) corpora before running `train.py`.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    # ---- Corpus roots -----------------------------------------------
    # Each root is expected to contain one sub-folder per class holding
    # .wav files, e.g. bc_root/Normal/*.wav, bc_root/Pain/*.wav, ...
    bc_root: str = "./data/baby_chillanto"
    dc_root: str = "./data/donateacry"

    bc_classes: Tuple[str, ...] = ("Normal", "Pain", "Hunger", "Deaf", "Asphyxia")
    dc_classes: Tuple[str, ...] = ("belly_pain", "hungry", "discomfort", "burping", "tired")

    # ---- Splits (Section III) ----------------------------------------
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    split_seed: int = 42

    # ---- Signal preprocessing (Section III-C) -------------------------
    sample_rate: int = 16_000
    segment_seconds: float = 1.0
    pre_emphasis: float = 0.97
    silence_rms_threshold: float = 1e-3

    # ---- Feature extraction (Section III-D, Eqs. 1-3) ------------------
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mels: int = 40
    n_mfcc: int = 13                 # + delta + delta-delta -> 39-D
    cqt_bins_per_octave: int = 24    # B = 24
    cqt_n_bins: int = 84             # 7 octaves x 12 (multiple of bins/oct)
    fmin: float = 32.7                # ~C1, standard CQT floor

    n_windows: int = 20               # T = 20 overlapping analysis windows / 1-s segment


@dataclass
class ModelConfig:
    # ---- Stream encoders (Section IV-A, Eqs. 4-7) ----------------------
    stream_channels: Tuple[int, ...] = (32, 64, 128, 128)
    se_reduction: int = 8
    embed_dim: int = 256              # d = 256

    # ---- Cross-Stream Tiny-Transformer fusion (Section IV-B, Eqs. 8-10) -
    cstf_layers: int = 2
    cstf_heads: int = 4
    cstf_ffn_dim: int = 512
    cstf_dropout: float = 0.1

    # ---- BiGRU temporal encoder (Section IV-C, Eqs. 11-15) -------------
    gru_hidden: int = 128             # x2 directions
    gru_layers: int = 1

    # ---- Domain-Adaptive Normalisation (Section IV-D, Eqs. 16-17) ------
    n_domains: int = 2                # {BC, DC}
    domain_embed_dim: int = 32

    # ---- Grad-CAM/Grad-CAM++-Guided Feature Gate (Section IV-E) --------
    ggfg_update_interval: int = 5     # K = 5 epochs
    ggfg_lambda: float = 0.5          # lambda in Eq. 23

    # ---- Classification heads -------------------------------------------
    n_classes_bc: int = 5
    n_classes_dc: int = 5
    classifier_dropout: float = 0.3


@dataclass
class TrainConfig:
    # ---- Optimisation (Section IV-F/G, TABLE 5) -------------------------
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4        # lambda_wd, Eq. 25
    warmup_epochs: int = 5
    epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10  # on validation macro-F1
    grad_clip_norm: float = 5.0

    # ---- Focal loss (Eq. 24) --------------------------------------------
    focal_gamma: float = 2.0

    # ---- Misc -------------------------------------------------------------
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"              # falls back to cpu automatically
    checkpoint_dir: str = "./checkpoints"
    log_every_n_steps: int = 20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = Config()
