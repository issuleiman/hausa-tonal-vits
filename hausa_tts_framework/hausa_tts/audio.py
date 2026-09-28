"""Audio front-end: linear spectrogram, mel filterbank, F0 extraction.

We deliberately avoid a hard dependency on librosa (heavy, and a frequent
source of version drift in low-resource stacks). Everything here is
numpy/torch only and matches the standard VITS preprocessing (a 1024-point
STFT, 256 hop, 80-bin mel, log magnitude).

``yin_f0`` is a compact YIN estimator (de Cheveigne & Kawahara 2002) used at
training time to compute the reference pitch contour that supervises both the
pitch predictor and the tonal adversarial critic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# mel / STFT
# ---------------------------------------------------------------------------


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_fft: int, n_mels: int = 80, fmin: float = 0.0, fmax: Optional[float] = None
) -> np.ndarray:
    """Slaney-style triangular mel filterbank, shape ``[n_mels, n_fft//2+1]``."""
    if fmax is None:
        fmax = sr / 2.0
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_bins)
    mel_points = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for k, f in enumerate(fft_freqs):
            if f < lo or f > hi:
                continue
            if f <= ce:
                fb[i, k] = (f - lo) / max(ce - lo, 1e-8)
            else:
                fb[i, k] = (hi - f) / max(hi - ce, 1e-8)
        norm = fb[i].sum()
        if norm > 0:
            fb[i] /= norm
    return fb


def dynamic_range_compression(x: np.ndarray, clip_val: float = 1e-5) -> np.ndarray:
    return np.log(np.clip(x, clip_val, None))


def linear_spectrogram(
    wav: np.ndarray, n_fft: int = 1024, hop: int = 256, win: int = 1024
) -> np.ndarray:
    """``[n_fft//2+1, T]`` linear magnitude spectrogram."""
    wav = np.asarray(wav, dtype=np.float32)
    pad = n_fft // 2   # centre padding, mirrors torch.stft(center=True)
    wav = np.pad(wav, (pad, pad), mode="reflect")
    window = np.hanning(win).astype(np.float32)
    n_frames = 1 + (len(wav) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        wav, shape=(n_frames, n_fft), strides=(wav.strides[0] * hop, wav.strides[0])
    ).copy()
    frames = frames * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return np.abs(spec).T.astype(np.float32)


def mel_spectrogram(
    wav: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256, win: int = 1024,
    n_mels: int = 80, fmin: float = 0.0, fmax: Optional[float] = None,
    fb: Optional[np.ndarray] = None,
) -> np.ndarray:
    if fb is None:
        fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    spec = linear_spectrogram(wav, n_fft, hop, win)
    mel = fb @ spec
    return dynamic_range_compression(mel)


# ---------------------------------------------------------------------------
# F0 (YIN)
# ---------------------------------------------------------------------------


def yin_f0(
    wav: np.ndarray,
    sr: int,
    hop: int = 256,
    win: int = 1024,
    fmin: float = 60.0,
    fmax: float = 500.0,
    threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-wise F0 (Hz) and voiced flag.

    Returns ``(f0, voiced)`` where ``f0`` is 0.0 for unvoiced frames.
    Suitable for Hausa: fmin 60 Hz covers the low male register, fmax 500 Hz
    covers the raised-low and question-finally raised high tones.
    """
    wav = np.asarray(wav, dtype=np.float64)
    tau_min = int(sr / fmax)
    tau_max = int(sr / fmin)
    n_frames = 1 + max(0, (len(wav) - win) // hop)
    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)
    if n_frames <= 0:
        return f0, voiced
    for i in range(n_frames):
        frame = wav[i * hop : i * hop + win]
        if frame.size < win:
            frame = np.pad(frame, (0, win - frame.size))
        d = np.zeros(tau_max, dtype=np.float64)
        for tau in range(1, tau_max):
            diff = frame[: win - tau] - frame[tau:]
            d[tau] = np.sum(diff * diff)
        cum = np.cumsum(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            cmnd = d[1:] * np.arange(1, tau_max) / np.clip(cum[1:], 1e-9, None)
        cmnd = np.nan_to_num(cmnd)
        tau_lo = max(1, tau_min)
        tau_hi = min(len(cmnd), tau_max - 1)
        if tau_hi <= tau_lo:
            continue
        cand = np.where(cmnd[tau_lo:tau_hi] < threshold)[0]
        if cand.size == 0:
            tau = tau_lo + int(np.argmin(cmnd[tau_lo:tau_hi]))
        else:
            tau = tau_lo + int(cand[0])
        if tau <= 0:
            continue
        f = sr / tau
        if fmin <= f <= fmax and cmnd[tau - 1] < 0.5:
            f0[i] = f
            voiced[i] = True
    return f0, voiced


def f0_to_continuous(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Convert Hz to a continuous (log) representation, interpolating unvoiced gaps.

    A continuous F0 track is what the pitch predictor consumes (as opposed to a
    quantised pitch class); it preserves the micro-prosody that carries Hausa
    downdrift and the phrase-final fall.
    """
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.asarray(voiced, dtype=bool)
    log_f0 = np.zeros_like(f0)
    if voiced.any():
        log_f0[voiced] = np.log(f0[voiced])
        idx = np.arange(len(f0))
        log_f0 = np.interp(idx, idx[voiced], log_f0[voiced]).astype(np.float32)
    return log_f0


def f0_stats(f0: np.ndarray, voiced: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """The 8 interpretable prosody statistics consumed by :class:`ProsodyEncoder`."""
    log_f0 = f0_to_continuous(f0, voiced)
    if voiced.sum() > 1:
        v = log_f0[voiced]
        mean, std = float(v.mean()), float(v.std())
        rng = float(np.percentile(v, 95) - np.percentile(v, 5))
        slope = float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 2 else 0.0
        jitter = float(np.abs(np.diff(f0[voiced])).mean())
    else:
        mean = std = rng = slope = jitter = 0.0
    return np.asarray(
        [mean, std, rng, float(voiced.mean()), float(energy.mean()),
         float(energy.std()), jitter, slope],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# harmonic supervision (PeriodVITS-style)
# ---------------------------------------------------------------------------


def harmonic_loss(f0: torch.Tensor, lin_spec: torch.Tensor, n_fft: int, sr: int,
                  n_harmonics: int = 40) -> torch.Tensor:
    """Encourage energy to sit at integer multiples of the predicted F0.

    ``f0``: ``[B, 1, T]`` Hz (0 for unvoiced). ``lin_spec``: ``[B, F, T]``
    linear magnitude. This is the differentiable "pitch is really there" signal
    that made PeriodVITS (Shirahata et al. 2023) strong on expressive speech;
    for Hausa it is what forces the vocoder to actually place the harmonics
    where the tone target says they should be.
    """
    b, _, t = f0.shape
    freqs = f0 / float(sr)  # cycles / sample
    bin_hz = sr / float(n_fft)
    idx = (freqs * n_harmonics * (n_fft / float(sr))).unsqueeze(-1)  # dummy to keep shape
    harm_idx = torch.arange(1, n_harmonics + 1, device=f0.device).view(1, 1, 1, -1)
    bins = (freqs.unsqueeze(-1) * harm_idx * float(n_fft)).round().long()  # [B,1,T,H]
    bins = bins.clamp(0, lin_spec.size(1) - 1)
    sel = torch.gather(
        lin_spec.unsqueeze(1).expand(-1, 1, -1, -1), 2,
        bins.permute(0, 1, 2, 3).reshape(b, 1, -1, 1).expand(-1, -1, -1, lin_spec.size(2))[:, :, :, :1]
        .reshape(b, 1, -1),
    )
    return -sel.mean()


def harmonic_pos_energy(f0: torch.Tensor, lin_spec: torch.Tensor, n_fft: int,
                        n_harmonics: int = 20) -> torch.Tensor:
    """Sum of harmonic magnitudes / total magnitude -- a *pitch confidence* index."""
    b, f_bins, t = lin_spec.shape
    freqs = f0  # Hz
    harm_idx = torch.arange(1, n_harmonics + 1, device=f0.device).view(1, 1, 1, -1)
    bins = (freqs.unsqueeze(-1) * harm_idx * (n_fft / f0.new_tensor(1.0)) / 1.0).long()
    bins = bins.clamp(0, f_bins - 1)
    flat = lin_spec.permute(0, 2, 1).reshape(b * t, f_bins)
    sel = torch.gather(flat.unsqueeze(1).expand(-1, n_harmonics, -1), 2, bins.reshape(b * t, -1).unsqueeze(1).expand(-1, n_harmonics, -1).permute(0, 2, 1)[:, :, :1].expand(-1, 1, -1)).squeeze(1)
    sel = torch.gather(flat, 1, bins.reshape(b * t, n_harmonics))
    return sel.sum(-1).view(b, t) / flat.sum(-1).clamp_min(1e-6).view(b, t)


# ---------------------------------------------------------------------------
# torch counterparts (used for reconstruction losses during training)
# ---------------------------------------------------------------------------


def linear_spectrogram_torch(y: torch.Tensor, n_fft: int = 1024, hop: int = 256,
                             win: int = 1024) -> torch.Tensor:
    """``[B, n_fft//2+1, T]`` magnitude spectrogram, centre padded (matches numpy)."""
    if y.dim() == 3:
        y = y.squeeze(1)
    spec = torch.stft(
        y, n_fft, hop_length=hop, win_length=win,
        window=torch.hann_window(win, device=y.device, periodic=False),
        center=True, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=True,
    )
    return spec.abs()


def mel_spectrogram_torch(y: torch.Tensor, sr: int, n_fft: int = 1024, hop: int = 256,
                          win: int = 1024, n_mels: int = 80, fmin: float = 0.0,
                          fmax: Optional[float] = None,
                          fb: Optional[np.ndarray] = None) -> torch.Tensor:
    """Log-mel of a batch of waveforms: ``[B, n_mels, T]``."""
    if fb is None:
        fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    fb_t = torch.from_numpy(np.asarray(fb, dtype=np.float32)).to(y.device)
    spec = linear_spectrogram_torch(y, n_fft, hop, win)
    mel = torch.matmul(fb_t, spec)
    return torch.log(torch.clamp(mel, min=1e-5))
