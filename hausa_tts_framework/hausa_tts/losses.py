"""Losses for the Hausa VITS extension.

Beyond the VITS objective (adversarial + feature matching + mel reconstruction
+ KL + duration), three Hausa-specific terms are added:

1. ``pitch_loss`` -- L1/L2 on the *continuous* F0 contour over voiced frames,
   which is what actually carries lexical tone.
2. ``tone_classification_loss`` -- a classifier maps the statistics of the
   synthesised F0 contour of each vowel to its lexical tone (H/L/F/R); its
   error rate is the tonal fidelity metric and its gradient shapes the pitch
   predictor.
3. ``tonal_adversarial_loss`` -- the tonal critic in :mod:`hausa_tts.vocoder`.

Plus an optional ``harmonic_loss`` term (PeriodVITS, Shirahata et al. 2023)
that pushes energy into the harmonics of the predicted F0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# basic adversarial losses
# ---------------------------------------------------------------------------


def discriminator_loss(d_real: List[torch.Tensor], d_fake: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(d_real, d_fake):
        r = r.view(-1)
        f = f.view(-1)
        loss = loss + torch.mean((1 - r) ** 2) + torch.mean(f**2)
    return loss / max(1, len(d_real))


def generator_loss(d_fake: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for f in d_fake:
        loss = loss + torch.mean((1 - f.view(-1)) ** 2)
    return loss / max(1, len(d_fake))


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = 0.0
    n = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + F.l1_loss(rl, gl.detach())
            n += 1
    return loss / max(1, n)


def _trim3(*tensors):
    """Trim several frame-level tensors to their common length along dim 2."""
    t = min(int(x.size(2)) for x in tensors)
    return [x[:, :, :t] for x in tensors]


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> torch.Tensor:
    """KL( q(z|y) || p(z|text) ) for diagonal Gaussians (VITS eq. 8)."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_p, m_p, logs_p, logs_q, z_mask = (
        lambda t: (t[0].float(), t[1].float(), t[2].float(), t[3].float(),
                   t[4][:, :, : t[0].size(2)]))(_trim3(z_p, m_p, logs_p, logs_q, z_mask))
    loss = logs_p - logs_q - 0.5
    loss = loss + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    loss = loss + 0.5 * torch.exp(2.0 * (logs_q - logs_p))
    return torch.sum(loss * z_mask) / (torch.sum(z_mask) * loss.size(1) + 1e-8)


def mel_reconstruction_loss(mel_gt: torch.Tensor, mel_pred: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(mel_pred, mel_gt)


# ---------------------------------------------------------------------------
# duration losses
# ---------------------------------------------------------------------------


def duration_loss_deterministic(pred_log: torch.Tensor, target_log: torch.Tensor,
                               x_mask: torch.Tensor) -> torch.Tensor:
    return (F.mse_loss(pred_log, target_log, reduction="none") * x_mask.squeeze(1)).sum() / (
        x_mask.sum() * pred_log.size(1) + 1e-8
    )


def duration_loss_stochastic(v: torch.Tensor, logdet: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    """NLL of the flow-transformed duration latent under a standard normal."""
    nll = 0.5 * (v**2).sum([1, 2]) - logdet
    return (nll / (x_mask.sum([1, 2]) + 1e-8)).mean()


# ---------------------------------------------------------------------------
# prosody losses
# ---------------------------------------------------------------------------


def _trim2(a: torch.Tensor, b: torch.Tensor):
    """Trim two frame-level tensors to their common length along dim 1."""
    t = min(int(a.size(1)), int(b.size(1)))
    return a[:, :t], b[:, :t]


def pitch_loss(f0_pred: torch.Tensor, f0_target: torch.Tensor, mask: torch.Tensor,
               voiced_mask: Optional[torch.Tensor] = None,
               loss_type: str = "l1") -> torch.Tensor:
    """Frame-level F0 loss in the (log) continuous domain."""
    f0_pred, f0_target = _trim2(f0_pred, f0_target)
    m = mask if voiced_mask is None else (mask * voiced_mask)
    m = m[:, : f0_pred.size(1)]
    diff = (f0_pred - f0_target).abs() if loss_type == "l1" else (f0_pred - f0_target) ** 2
    return (diff * m).sum() / (m.sum() + 1e-8)


def energy_loss(e_pred: torch.Tensor, e_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    e_pred, e_target = _trim2(e_pred, e_target)
    mask = mask[:, : e_pred.size(1)] if mask.dim() == 2 else mask[:, :, : e_pred.size(1)]
    return ((e_pred - e_target).abs() * mask).sum() / (mask.sum() + 1e-8)


def prosody_stats_loss(stats_pred: torch.Tensor, stats_target: torch.Tensor) -> torch.Tensor:
    """Supervise the utterance-level prosody statistics (scale-invariant)."""
    scale = stats_target.abs().mean() + 1e-6
    return F.smooth_l1_loss(stats_pred / scale, stats_target / scale)


# ---------------------------------------------------------------------------
# tonal losses
# ---------------------------------------------------------------------------


def vowel_pooling(values: torch.Tensor, value_mask: torch.Tensor,
                  seg_ids: torch.Tensor, n_seg: int):
    """Pool a frame-level track over vowel segments.

    Returns ``(mean, slope, rng, dur, valid)`` per segment, all ``[B, n_seg]``.
    """
    # the alignment grid, the F0 track and the vowel mask can differ by a frame;
    # pooling is only meaningful over the frames all three agree on.
    T = min(values.size(1), seg_ids.size(1))
    values = values[:, :T]
    value_mask = value_mask[:, :T]
    seg_ids = seg_ids[:, :T]
    b = int(values.size(0))
    device = values.device
    mean = values.new_zeros(b, n_seg)
    slope = values.new_zeros(b, n_seg)
    rng = values.new_zeros(b, n_seg)
    dur = values.new_zeros(b, n_seg)
    valid = torch.zeros(b, n_seg, dtype=torch.bool, device=device)
    for bi in range(b):
        ids = seg_ids[bi]
        m = value_mask[bi]
        for s in range(n_seg):
            sel = (ids == s) & m
            n = int(sel.sum().item())
            if n < 2:
                continue
            vals = values[bi][sel]
            mean[bi, s] = vals.mean()
            dur[bi, s] = float(n)
            rng[bi, s] = vals.max() - vals.min()
            idx = torch.arange(n, device=device).float()
            idx = idx - idx.mean()
            denom = (idx**2).sum().clamp_min(1e-6)
            slope[bi, s] = ((vals - vals.mean()) * idx).sum() / denom
            valid[bi, s] = True
    return mean, slope, rng, dur, valid


def tone_classification_loss(
    f0_pred: torch.Tensor,
    f0_mask: torch.Tensor,
    seg_ids: torch.Tensor,
    tone_labels: torch.Tensor,
    tone_valid: torch.Tensor,
    classifier,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify the tone of each vowel from its synthesised F0 contour.

    ``seg_ids`` labels each frame with the vowel index it belongs to (-1 for
    consonants/pauses). Returns ``(loss, accuracy, n_scored)``.
    """
    n_seg = seg_ids.max().item() + 1 if seg_ids.numel() else 0
    if n_seg <= 0:
        zero = f0_pred.new_zeros(())
        return zero, zero, torch.zeros((), device=f0_pred.device)
    mean, slope, rng, dur, valid = vowel_pooling(f0_pred, f0_mask, seg_ids, n_seg)
    dnorm = torch.log(dur.clamp_min(1.0))
    feats = torch.stack([mean, slope * 10.0, rng, dnorm, mean * dnorm], dim=-1)
    logits = classifier(feats)
    target = tone_labels.clamp(0, 3)
    mask = valid & tone_valid.bool()
    if mask.sum() == 0:
        zero = f0_pred.new_zeros(())
        return zero, zero, torch.zeros((), device=f0_pred.device)
    loss = F.cross_entropy(logits[mask], target[mask])
    acc = (logits[mask].argmax(-1) == target[mask]).float().mean()
    return loss, acc, mask.sum().float()


def tonal_adversarial_generator(d_fake: List[torch.Tensor]) -> torch.Tensor:
    return generator_loss(d_fake)


# ---------------------------------------------------------------------------
# harmonic (PeriodVITS-style) loss
# ---------------------------------------------------------------------------


def harmonic_loss(f0_hz: torch.Tensor, lin_spec: torch.Tensor, n_fft: int,
                  sr: int, n_harmonics: int = 30) -> torch.Tensor:
    """Push spectral energy onto the integer multiples of the predicted F0.

    ``f0_hz``: ``[B, T]`` Hz (0 where unvoiced). ``lin_spec``: ``[B, F, T]``
    linear magnitude. Both are trimmed to their common frame count, so a
    one-frame difference between the vocoder output and the analysis window can
    never crash training. Differentiable w.r.t. F0 and the spectrogram.
    """
    if f0_hz.dim() == 3:
        f0_hz = f0_hz.squeeze(1)
    b, f_bins, t_spec = lin_spec.shape
    t = min(int(t_spec), int(f0_hz.size(1)))
    if t == 0:
        return lin_spec.new_zeros(())
    f0 = f0_hz[:, :t]
    lin = lin_spec[:, :, :t]
    voiced = (f0 > 1.0).float()
    harm = torch.arange(1, n_harmonics + 1, device=f0.device).view(1, 1, -1)
    bins = (f0.unsqueeze(-1) * harm * (n_fft / float(sr))).round().long()
    bins = bins.clamp(0, f_bins - 1)
    flat = lin.permute(0, 2, 1).reshape(-1, f_bins)                  # [B*T, F]
    gathered = torch.gather(flat, 1, bins.reshape(-1, n_harmonics))   # [B*T, H]
    total = flat.sum(-1).clamp_min(1e-6)
    ratio = (gathered.sum(-1) / total).view(b, t)
    loss = -torch.log(ratio.clamp_min(1e-6) + 1e-6)
    return (loss * voiced).sum() / (voiced.sum() + 1e-8)


def pitch_confidence(f0_hz: torch.Tensor, lin_spec: torch.Tensor, n_fft: int,
                     sr: int, n_harmonics: int = 30) -> torch.Tensor:
    """Harmonic share per frame (diagnostic, no gradient needed)."""
    with torch.no_grad():
        b, f_bins, t = lin_spec.shape
        harm = torch.arange(1, n_harmonics + 1, device=f0_hz.device).view(1, 1, -1)
        bins = (f0_hz.unsqueeze(-1) * harm * (n_fft / float(sr))).round().long().clamp(0, f_bins - 1)
        flat = lin_spec.permute(0, 2, 1).reshape(b * t, f_bins)
        gathered = torch.gather(flat, 1, bins.reshape(b * t, n_harmonics))
        return (gathered.sum(-1) / flat.sum(-1).clamp_min(1e-6)).view(b, t)
