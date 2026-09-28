"""Monotonic alignment search (MAS) and alignment utilities.

MAS is what lets VITS learn text<->speech alignment without an external aligner
(Kim et al. 2021). The implementation here is an explicit, dependency-free
dynamic program over ``[T_text, T_spec]`` so it is easy to audit and to extend.

Two Hausa-specific additions:

* an optional anchor prior (:func:`attention_log_prior`) that encourages
  vowels to own frames and discourages pause tokens from swallowing voiced
  audio -- the dominant alignment failure on read-speech corpora with short
  pauses;
* :func:`durations_from_alignment` plus :func:`length_aware_floor` that
  guarantee a *minimum* frame budget for long vowels, which stabilises the
  first training epochs on small corpora where the length contrast is easy to
  collapse.

An external aligner (e.g. the Montreal Forced Aligner, McAuliffe et al.) can be
used instead: the loader accepts pre-computed per-phoneme durations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# MAS
# ---------------------------------------------------------------------------


def _masked_path(cost: np.ndarray) -> np.ndarray:
    """Monotone path maximising ``sum(cost[i, j])`` for a single item.

    ``cost``: ``[T_text, T_spec]``. Returns a 0/1 matrix of the same shape.
    Constraints enforced: every text token owns at least one frame, every frame
    belongs to exactly one token, and the path is monotone non-decreasing.
    """
    t_txt, t_spec = cost.shape
    neg_inf = -1e9
    value = np.full((t_txt, t_spec), neg_inf, dtype=np.float64)
    back = np.zeros((t_txt, t_spec), dtype=np.int8)  # 0 = came from (i, j-1), 1 = (i-1, j-1)
    for j in range(t_spec):
        i_min = 0
        i_max = min(t_txt - 1, j)
        # feasibility: enough remaining frames for the remaining tokens
        i_min = max(i_min, t_txt - 1 - (t_spec - 1 - j))
        if i_min > i_max:
            continue
        for i in range(i_min, i_max + 1):
            c = cost[i, j]
            if i == 0 and j == 0:
                value[i, j] = c
                back[i, j] = 0
                continue
            best, arg = neg_inf, 0
            if j > 0 and value[i, j - 1] > neg_inf:
                best, arg = value[i, j - 1], 0
            if i > 0 and j > 0 and value[i - 1, j - 1] > best:
                best, arg = value[i - 1, j - 1], 1
            if best <= neg_inf / 2:
                return np.zeros((t_txt, t_spec), dtype=np.float32)
            value[i, j] = c + best
            back[i, j] = arg
    path = np.zeros((t_txt, t_spec), dtype=np.float32)
    i, j = t_txt - 1, t_spec - 1
    while True:
        path[i, j] = 1.0
        if i == 0 and j == 0:
            break
        if back[i, j] == 1:
            i, j = i - 1, j - 1
        else:
            j = j - 1
        if i < 0 or j < 0:
            break
    return path


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Batched MAS.

    Args:
        neg_cent: ``[B, T_text, T_spec]`` costs (higher is better).
        mask: ``[B, T_text, T_spec]`` validity mask.
    """
    device, dtype = neg_cent.device, neg_cent.dtype
    cost = (neg_cent * mask).detach().cpu().numpy().astype(np.float64)
    m = mask.detach().cpu().numpy().astype(bool)
    b, t_txt, t_spec = cost.shape
    out = np.zeros((b, t_txt, t_spec), dtype=np.float32)
    for bi in range(b):
        tt = int(m[bi].any(axis=1).sum())
        ts = int(m[bi].any(axis=0).sum())
        if tt == 0 or ts == 0:
            continue
        sub = cost[bi, :tt, :ts].copy()
        sub[~m[bi, :tt, :ts]] = -1e9
        out[bi, :tt, :ts] = _masked_path(sub)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build the alignment matrix from integer frame durations.

    ``duration``: ``[B, T_text]``, ``mask``: ``[B, 1, T_spec]``.
    """
    b, t_txt = duration.size()
    t_spec = mask.size(2)
    path = torch.zeros(b, t_txt, t_spec, device=duration.device)
    for bi in range(b):
        start = 0
        for j in range(t_txt):
            d = int(duration[bi, j].item())
            if d <= 0:
                continue
            end = min(start + d, t_spec)
            if end > start:
                path[bi, j, start:end] = 1.0
            start = end
    return path


def attention_log_prior(
    neg_cent: torch.Tensor, vowel_mask: torch.Tensor, pause_mask: torch.Tensor,
    vowel_bonus: float = 0.05, pause_penalty: float = 0.05,
) -> torch.Tensor:
    return neg_cent + vowel_bonus * vowel_mask.unsqueeze(-1) - pause_penalty * pause_mask.unsqueeze(-1)


def durations_from_alignment(path: torch.Tensor, x_mask: torch.Tensor,
                             y_mask: torch.Tensor) -> torch.Tensor:
    return (path * y_mask.unsqueeze(1)).sum(-1) * x_mask.squeeze(1)


def length_aware_floor(durations: torch.Tensor, length_ids: torch.Tensor,
                       x_mask: torch.Tensor, long_min: int = 3, short_min: int = 1) -> torch.Tensor:
    """Minimum frame budget per phoneme: long vowels get more frames than short ones.

    Hausa long vowels are roughly 1.7-2.0x short ones; enforcing this floor
    early in training prevents the length contrast from being averaged away.
    """
    floor = torch.where(length_ids == 2, torch.full_like(durations, long_min),
                        torch.full_like(durations, short_min)).float()
    return torch.maximum(durations.float(), floor) * x_mask.squeeze(1)


def expand_priors(z: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """Frame-level prior: ``[B, C, T_text] x [B, T_text, T_spec] -> [B, C, T_spec]``."""
    return torch.matmul(z, path)


def mask_from_path(path: torch.Tensor) -> torch.Tensor:
    """Derive the frame mask implied by an alignment."""
    return (path.sum(1) > 0).unsqueeze(1).float()
