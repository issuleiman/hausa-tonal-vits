"""Evaluation metrics for tonal TTS.

Aggregates the metrics the TTS evaluation literature actually relies on
(Cooper et al. 2024; Galdino & Matos 2025; Yang et al. 2025) and adds two
tonal ones that are specific to this work:

* ``tone_accuracy``  -- fraction of vowels whose synthesised F0 contour is
  classified as the intended lexical tone (H/L/F/Raised-L). This is the
  metric that a waveform-only MOS cannot see.
* ``tone_faithfulness`` -- correlation between the predicted and the intended
  tone *pattern* over an utterance, i.e. a reference-free tonal metric in the
  spirit of DunDun (Daudu et al. 2026) and ViTone (Phan & Ha 2026).

Plus the classical set: MCD, log-F0 RMSE / correlation, duration MAE,
speaker-embedding cosine similarity, and RTF.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from . import audio as haudio


# ---------------------------------------------------------------------------
# spectral / prosodic
# ---------------------------------------------------------------------------


def mel_cepstral_distortion(mel_a: np.ndarray, mel_b: np.ndarray, n_ceps: int = 25) -> float:
    """MCD in dB between two log-mel/cepstral sequences (lower is better)."""
    a, b = np.asarray(mel_a), np.asarray(mel_b)
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    cep_a = _dct(a, n_ceps)
    cep_b = _dct(b, n_ceps)
    diff = cep_a - cep_b
    diff[0] = 0.0
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt((diff**2).sum(axis=0))))


def _dct(x: np.ndarray, n_ceps: int) -> np.ndarray:
    m, t = x.shape
    k = np.arange(n_ceps).reshape(-1, 1)
    n = np.arange(m).reshape(1, -1)
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * m))
    return basis @ x


def log_f0_rmse(f0_a: np.ndarray, f0_b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a, b = np.asarray(f0_a, dtype=np.float64), np.asarray(f0_b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if mask is None:
        mask = (a > 1.0) & (b > 1.0)
    else:
        mask = np.asarray(mask)[:n].astype(bool) & (a > 1.0) & (b > 1.0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((np.log(a[mask]) - np.log(b[mask])) ** 2)))


def f0_correlation(f0_a: np.ndarray, f0_b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a, b = np.asarray(f0_a, dtype=np.float64), np.asarray(f0_b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if mask is None:
        mask = (a > 1.0) & (b > 1.0)
    else:
        mask = np.asarray(mask)[:n].astype(bool) & (a > 1.0) & (b > 1.0)
    if mask.sum() < 3:
        return float("nan")
    x, y = np.log(a[mask]), np.log(b[mask])
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def duration_mae(pred: Sequence[int], target: Sequence[int]) -> float:
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    n = min(len(p), len(t))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(p[:n] - t[:n])))


def length_ratio_error(gen_f0_voiced_lens: Sequence[int], ref: Sequence[int]) -> float:
    """Error on the long/short vowel duration ratio (Hausa: expect ~1.7-2.0)."""
    if len(ref) < 2:
        return float("nan")
    return float(abs(np.mean(gen_f0_voiced_lens) / max(np.mean(ref), 1e-6) - 1.0))


# ---------------------------------------------------------------------------
# tonal
# ---------------------------------------------------------------------------


def tone_accuracy(tone_pred: Sequence[int], tone_true: Sequence[int],
                  valid: Optional[Sequence[bool]] = None) -> float:
    p, t = np.asarray(tone_pred), np.asarray(tone_true)
    n = min(len(p), len(t))
    if n == 0:
        return float("nan")
    if valid is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)[:n]
    v = valid & (t[:n] != 0)
    if v.sum() == 0:
        return float("nan")
    return float((p[:n][v] == t[:n][v]).mean())


def tone_faithfulness(f0_hz: np.ndarray, tone_ids: np.ndarray,
                      vowel_positions: Sequence[Tuple[int, int]],
                      levels: Optional[Sequence[float]] = None) -> float:
    """Correlation between the intended tone levels and the produced means.

    ``vowel_positions``: list of ``(start, end)`` frame ranges, one per vowel;
    ``tone_ids``: the intended tone id per vowel.
    """
    levels = levels if levels is not None else [0.0, 0.0, -3.0, -1.0, -1.6]
    means = []
    want = []
    for (s, e), t in zip(vowel_positions, tone_ids):
        seg = f0_hz[s:e]
        voiced = seg > 1.0
        if voiced.sum() < 2:
            continue
        means.append(float(np.log(seg[voiced]).mean()))
        want.append(float(levels[int(t)]))
    if len(means) < 3:
        return float("nan")
    m, w = np.asarray(means), np.asarray(want)
    if m.std() < 1e-9 or w.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(m, w)[0, 1])


# ---------------------------------------------------------------------------
# speaker
# ---------------------------------------------------------------------------


def speaker_similarity(enc, mel_a: torch.Tensor, mel_b: torch.Tensor) -> float:
    with torch.no_grad():
        a = enc(mel_a.unsqueeze(0).transpose(1, 2))
        b = enc(mel_b.unsqueeze(0).transpose(1, 2))
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return float((a * b).sum(-1).item())


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def evaluate_pair(
    gen_wav: np.ndarray, ref_wav: np.ndarray, sr: int, cfg,
    *, tone_pred: Optional[Sequence[int]] = None,
    tone_true: Optional[Sequence[int]] = None,
    vowel_positions: Optional[Sequence[Tuple[int, int]]] = None,
    speaker_enc=None,
) -> Dict[str, float]:
    """Full metric set for one synthesised / reference pair."""
    fb = haudio.mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    mel_g = haudio.mel_spectrogram(gen_wav, sr, cfg.n_fft, cfg.hop_length, cfg.win_length,
                                   cfg.n_mels, cfg.fmin, cfg.fmax, fb=fb)
    mel_r = haudio.mel_spectrogram(ref_wav, sr, cfg.n_fft, cfg.hop_length, cfg.win_length,
                                   cfg.n_mels, cfg.fmin, cfg.fmax, fb=fb)
    f0_g, v_g = haudio.yin_f0(gen_wav, sr, cfg.hop_length, cfg.win_length, cfg.f0_min_hz, cfg.f0_max_hz)
    f0_r, v_r = haudio.yin_f0(ref_wav, sr, cfg.hop_length, cfg.win_length, cfg.f0_min_hz, cfg.f0_max_hz)
    mask = v_g & v_r
    out = {
        "mcd_db": mel_cepstral_distortion(mel_g, mel_r),
        "f0_rmse_log": log_f0_rmse(f0_g, f0_r, mask),
        "f0_corr": f0_correlation(f0_g, f0_r, mask),
        "voiced_ratio_gen": float(v_g.mean()),
        "voiced_ratio_ref": float(v_r.mean()),
        "dur_ratio": float(len(gen_wav) / max(1, len(ref_wav))),
    }
    if tone_pred is not None and tone_true is not None:
        out["tone_accuracy"] = tone_accuracy(tone_pred, tone_true)
    if vowel_positions and tone_true is not None and len(tone_true) > 0:
        out["tone_faithfulness"] = tone_faithfulness(f0_g, tone_true, vowel_positions)
    if speaker_enc is not None:
        try:
            out["spk_sim"] = speaker_similarity(
                speaker_enc, torch.from_numpy(mel_g).float(), torch.from_numpy(mel_r).float()
            )
        except Exception:
            pass
    return out


def real_time_factor(audio_seconds: float, wall_seconds: float) -> float:
    return float(wall_seconds / max(audio_seconds, 1e-6))
