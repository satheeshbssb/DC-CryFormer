"""
features.py
-----------
Implements the three acoustically complementary descriptors of Section
III-D: MFCC (Eq. 1), log-Mel spectrogram (Eq. 2), and a joint Constant-Q /
Tonnetz representation (Eq. 3). Each descriptor is computed independently
and kept at its own native time-frequency resolution -- they are never
resized or stacked into a single composite image, per Section III-D.
"""
import numpy as np
import librosa

from config import DataConfig


def load_and_preprocess(path: str, cfg: DataConfig) -> np.ndarray:
    """Load a waveform, resample, pre-emphasise and peak-normalise it
    (Section III-C)."""
    y, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    y = librosa.effects.preemphasis(y, coef=cfg.pre_emphasis)
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak
    return y.astype(np.float32)


def segment_signal(y: np.ndarray, cfg: DataConfig):
    """Split a waveform into non-overlapping 1-second segments, discarding
    near-silent frames (Section III-C)."""
    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    n_segs = len(y) // seg_len
    segments = []
    for i in range(n_segs):
        seg = y[i * seg_len:(i + 1) * seg_len]
        rms = np.sqrt(np.mean(seg ** 2) + 1e-12)
        if rms >= cfg.silence_rms_threshold:
            segments.append(seg)
    return segments


def extract_mfcc(y: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Eq. 1: 13 MFCCs + delta + delta-delta -> 39-D per frame.
    Returns array of shape (39, T_frames)."""
    win_length = int(cfg.win_length_ms * cfg.sample_rate / 1000)
    hop_length = int(cfg.hop_length_ms * cfg.sample_rate / 1000)
    n_fft = int(2 ** np.ceil(np.log2(win_length)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=cfg.sample_rate, n_mfcc=cfg.n_mfcc, n_mels=cfg.n_mels,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window="hamming",
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate([mfcc, delta, delta2], axis=0)  # (39, T)
    return feats.astype(np.float32)


def extract_logmel(y: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Eq. 2: log-compressed Mel filter-bank energy map.
    Returns array of shape (n_mels, T_frames)."""
    win_length = int(cfg.win_length_ms * cfg.sample_rate / 1000)
    hop_length = int(cfg.hop_length_ms * cfg.sample_rate / 1000)
    n_fft = int(2 ** np.ceil(np.log2(win_length)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_mels=cfg.n_mels,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window="hamming", power=2.0,
    )
    log_mel = np.log1p(mel)  # log(1 + |X|^2 . H_m)
    return log_mel.astype(np.float32)


def extract_cqt_tonnetz(y: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Eq. 3: Constant-Q Transform magnitude, projected onto the
    6-D Tonnetz harmonic-network space via a chroma-CQT intermediate.
    Returns array of shape (6, T_frames)."""
    hop_length = int(cfg.hop_length_ms * cfg.sample_rate / 1000)

    chroma_cqt = librosa.feature.chroma_cqt(
        y=y, sr=cfg.sample_rate, hop_length=hop_length,
        fmin=cfg.fmin, n_chroma=12,
        bins_per_octave=cfg.cqt_bins_per_octave,
    )
    tonnetz = librosa.feature.tonnetz(chroma=chroma_cqt, sr=cfg.sample_rate)
    return tonnetz.astype(np.float32)  # (6, T)


def extract_all_streams(y_segment: np.ndarray, cfg: DataConfig):
    """Compute the three full-segment feature maps and split each into
    cfg.n_windows overlapping analysis windows (Section IV-C: "a one-second
    segment is itself divided into T=20 overlapping analysis windows prior
    to stream encoding"). Returns three arrays of shape
    (n_windows, n_freq_bins, win_frames)."""
    mfcc = extract_mfcc(y_segment, cfg)          # (39, T_total)
    logmel = extract_logmel(y_segment, cfg)      # (n_mels, T_total)
    tonnetz = extract_cqt_tonnetz(y_segment, cfg)  # (6, T_total)

    mfcc_w = windowize(mfcc, cfg.n_windows)
    logmel_w = windowize(logmel, cfg.n_windows)
    tonnetz_w = windowize(tonnetz, cfg.n_windows)
    return mfcc_w, logmel_w, tonnetz_w


def windowize(feat: np.ndarray, n_windows: int) -> np.ndarray:
    """Split a (F, T) feature map into n_windows overlapping (50%) windows
    of equal length along the time axis, returning (n_windows, F, win_frames).
    Short inputs are edge-padded so every window is fully populated."""
    f_dim, t_dim = feat.shape
    win_frames = max(2, int(np.ceil(2 * t_dim / (n_windows + 1))))
    hop_frames = max(1, win_frames // 2)
    required_len = hop_frames * (n_windows - 1) + win_frames

    if t_dim < required_len:
        feat = np.pad(feat, ((0, 0), (0, required_len - t_dim)), mode="edge")

    windows = np.empty((n_windows, f_dim, win_frames), dtype=feat.dtype)
    for i in range(n_windows):
        start = i * hop_frames
        windows[i] = feat[:, start:start + win_frames]
    return windows


def normalise(feat: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation per feature map (fit
    statistics on the training split and reuse them at inference in a
    real pipeline; see dataset.py for a simple running-stats example)."""
    mu, sigma = feat.mean(), feat.std() + 1e-8
    return (feat - mu) / sigma
