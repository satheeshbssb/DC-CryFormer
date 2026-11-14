
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

    # ---- Splits ----------------------------------------
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    split_seed: int = 42

    # ---- Signal preprocessing -------------------------
    sample_rate: int = 16_000
    segment_seconds: float = 1.0
    pre_emphasis: float = 0.97
    silence_rms_threshold: float = 1e-3

    # ---- Feature extraction------------------
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mels: int = 40
    n_mfcc: int = 13                 
    cqt_bins_per_octave: int = 24    
    cqt_n_bins: int = 84             
    fmin: float = 32.7                

    n_windows: int = 20               


@dataclass
class ModelConfig:
    # ---- Stream encoders ----------------------
    stream_channels: Tuple[int, ...] = (32, 64, 128, 128)
    se_reduction: int = 8
    embed_dim: int = 256              

    # ---- Cross-Stream Tiny-Transformer fusion-----------
    cstf_layers: int = 2
    cstf_heads: int = 4
    cstf_ffn_dim: int = 512
    cstf_dropout: float = 0.1

    # ---- BiGRU temporal encoder -------------
    gru_hidden: int = 128             
    gru_layers: int = 1

    # ---- Domain-Adaptive Normalisation  ------
    n_domains: int = 2                
    domain_embed_dim: int = 32

    # ---- Grad-CAM/Grad-CAM++-Guided Feature Gate--------
    ggfg_update_interval: int = 5     
    ggfg_lambda: float = 0.5          

    # ---- Classification heads -------------------------------------------
    n_classes_bc: int = 5
    n_classes_dc: int = 5
    classifier_dropout: float = 0.3


@dataclass
class TrainConfig:
    # ---- Optimisation -------------------------
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4       
    warmup_epochs: int = 5
    epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10  
    grad_clip_norm: float = 5.0

    # ---- Focal loss --------------------------------------------
    focal_gamma: float = 2.0

    # ---- Misc -------------------------------------------------------------
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"             
    checkpoint_dir: str = "./checkpoints"
    log_every_n_steps: int = 20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = Config()
