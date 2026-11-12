"""
dataset.py
----------
PyTorch Dataset for Baby Chillanto (BC) and Donate-a-Cry (DC), implementing
the 70/15/15 patient/file-wise split of Section III-B and the feature
pipeline of features.py. Both corpora are exposed through the same class so
that a single collate_fn and training loop can serve either domain, per
Section IV-D (Domain-Adaptive Normalisation).
"""
import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DataConfig
import features as F

DOMAIN_BC = 0
DOMAIN_DC = 1


@dataclass
class Sample:
    path: str
    label: int
    domain: int


def _list_files(root: str, classes: Tuple[str, ...]) -> List[Tuple[str, int]]:
    """Recursively collect (filepath, label_index) pairs, one sub-folder
    per class, e.g. root/Normal/*.wav."""
    items = []
    for label_idx, cls in enumerate(classes):
        pattern = os.path.join(root, cls, "*.wav")
        for p in sorted(glob.glob(pattern)):
            items.append((p, label_idx))
    if not items:
        raise FileNotFoundError(
            f"No .wav files found under {root} for classes {classes}. "
            "Point DataConfig.bc_root / dc_root at your local corpus copies."
        )
    return items


def make_splits(root: str, classes: Tuple[str, ...], domain: int, cfg: DataConfig,
                 seed: int) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Stratified per-class 70/15/15 split, file-wise (Section III-B). For
    Baby Chillanto, group by patient ID prefix instead if your file naming
    encodes it, to guarantee no patient straddles two splits."""
    rng = random.Random(seed)
    items = _list_files(root, classes)

    by_class = {}
    for path, label in items:
        by_class.setdefault(label, []).append(path)

    train, val, test = [], [], []
    for label, paths in by_class.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * cfg.train_split))
        n_val = int(round(n * cfg.val_split))
        train += [Sample(p, label, domain) for p in paths[:n_train]]
        val += [Sample(p, label, domain) for p in paths[n_train:n_train + n_val]]
        test += [Sample(p, label, domain) for p in paths[n_train + n_val:]]
    return train, val, test


class CryDataset(Dataset):
    """Segments every waveform into 1-second frames at construction time
    (small corpora: BC ~2.3k and DC ~2.0k samples fit comfortably in
    memory-mapped form). For larger corpora, switch to lazy per-item
    segment extraction in __getitem__."""

    def __init__(self, samples: List[Sample], cfg: DataConfig, augment: bool = False):
        self.cfg = cfg
        self.augment = augment
        self.index: List[Tuple[str, int, int, int]] = []  # (path, label, domain, seg_idx)
        self._cache = {}

        for s in samples:
            y = F.load_and_preprocess(s.path, cfg)
            segs = F.segment_signal(y, cfg)
            if not segs:
                continue
            self._cache[s.path] = segs
            for seg_idx in range(len(segs)):
                self.index.append((s.path, s.label, s.domain, seg_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, label, domain, seg_idx = self.index[i]
        seg = self._cache[path][seg_idx].copy()

        if self.augment:
            seg = _augment_waveform(seg, self.cfg)

        mfcc, logmel, tonnetz = F.extract_all_streams(seg, self.cfg)
        mfcc = F.normalise(mfcc)
        logmel = F.normalise(logmel)
        tonnetz = F.normalise(tonnetz)

        return {
            "mfcc": torch.from_numpy(mfcc).unsqueeze(1),        # (T, 1, F_mfcc, win)
            "logmel": torch.from_numpy(logmel).unsqueeze(1),    # (T, 1, F_mel, win)
            "tonnetz": torch.from_numpy(tonnetz).unsqueeze(1),  # (T, 1, F_tonnetz, win)
            "label": torch.tensor(label, dtype=torch.long),
            "domain": torch.tensor(domain, dtype=torch.long),
        }


def _augment_waveform(y: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Light augmentation: additive Gaussian noise + random gain, applied
    only to the training split. Extend with time/pitch shifting as needed."""
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.005, size=y.shape).astype(np.float32)
        y = y + noise
    if random.random() < 0.5:
        gain = random.uniform(0.85, 1.15)
        y = y * gain
    return np.clip(y, -1.0, 1.0)


def build_datasets(cfg: DataConfig):
    """Returns a dict of 6 datasets: {bc,dc} x {train,val,test}."""
    bc_train_s, bc_val_s, bc_test_s = make_splits(
        cfg.bc_root, cfg.bc_classes, DOMAIN_BC, cfg, cfg.split_seed)
    dc_train_s, dc_val_s, dc_test_s = make_splits(
        cfg.dc_root, cfg.dc_classes, DOMAIN_DC, cfg, cfg.split_seed)

    return {
        "bc_train": CryDataset(bc_train_s, cfg, augment=True),
        "bc_val": CryDataset(bc_val_s, cfg, augment=False),
        "bc_test": CryDataset(bc_test_s, cfg, augment=False),
        "dc_train": CryDataset(dc_train_s, cfg, augment=True),
        "dc_val": CryDataset(dc_val_s, cfg, augment=False),
        "dc_test": CryDataset(dc_test_s, cfg, augment=False),
    }
