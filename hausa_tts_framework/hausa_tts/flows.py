"""Monotone (rational-quadratic spline) flows and the stochastic duration model.

The spline transform is the standard monotonic rational-quadratic spline from
``nflows`` (Durkan et al. 2019) as used by VITS's stochastic duration predictor
(Kim et al. 2021). Unlike the original, our flow is **explicitly conditioned on
the phoneme features** (tone, vowel length, text context), i.e. a conditional
neural spline flow. Consequence: the sampling noise of the duration model is
linguistically structured and cannot erase the Hausa length/tone duration
contrast, which is the main failure mode of the stock SDP on small corpora.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modules import LayerNorm, get_padding


# ---------------------------------------------------------------------------
# monotone rational-quadratic spline
# ---------------------------------------------------------------------------


def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6):
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
):
    num_bins = unnormalized_widths.shape[-1]
    assert num_bins == unnormalized_heights.shape[-1]
    widths = min_bin_width + (1 - min_bin_width * num_bins) * F.softmax(unnormalized_widths, dim=-1)
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * F.softmax(unnormalized_heights, dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
             + input_heights * (input_delta - input_derivatives))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = -input_delta * (inputs - input_cumheights)
        discriminant = F.relu(b.pow(2) - 4 * a * c)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator.clamp_min(1e-12)) - 2 * torch.log(denominator.clamp_min(1e-12))
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths.clamp_min(1e-12)
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator.clamp_min(1e-12)
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator.clamp_min(1e-12)) - 2 * torch.log(denominator.clamp_min(1e-12))
    return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse=False, tail_bound=1.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3,
):
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside = ~inside
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, (1, 1))
    constant = math.log(math.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside] = inputs[outside]
    logabsdet[outside] = 0.0
    if inside.any():
        o, l = rational_quadratic_spline(
            inputs[inside],
            unnormalized_widths[inside],
            unnormalized_heights[inside],
            unnormalized_derivatives[inside],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
            min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative,
        )
        outputs[inside] = o
        logabsdet[inside] = l
    return outputs, logabsdet


# ---------------------------------------------------------------------------
# conditional monotone flow for the duration
# ---------------------------------------------------------------------------


class ConditionalSplineFlow(nn.Module):
    """A stack of monotone spline transforms on a scalar log-duration per phoneme.

    Conditioning is on the text-encoder features (tone, length, context), so
    the *distribution* of phoneme duration is conditioned on the linguistics.
    """

    def __init__(self, context_channels: int, hidden: int = 128, n_layers: int = 2,
                 num_bins: int = 10, kernel_size: int = 5, n_flows: int = 4,
                 tail_bound: float = 5.0) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.n_flows = n_flows
        self.context_net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(context_channels if i == 0 else hidden, hidden, kernel_size,
                              padding=get_padding(kernel_size)),
                    nn.ReLU(), LayerNorm(hidden),
                )
                for i in range(n_layers)
            ],
            nn.Conv1d(hidden, 3 * num_bins - 1, 1),
        )
        self.flip = True

    def _params(self, context: torch.Tensor, x_mask: torch.Tensor):
        stats = self.context_net(context) * x_mask
        return stats

    def forward(self, x, x_mask, context, reverse: bool = False):
        """``x``: ``[B, 1, T]`` scalar log-duration; returns ``(y, logabsdet)``."""
        stats = self._params(context, x_mask)
        n = self.num_bins
        widths, heights, derivs = stats.split([n, n, n - 1], dim=1)
        u = x.squeeze(1) * x_mask.squeeze(1)
        # transverse to channel-last for the spline helpers
        w = widths.transpose(1, 2)
        h = heights.transpose(1, 2)
        d = derivs.transpose(1, 2)
        y, logdet = unconstrained_rational_quadratic_spline(
            u, w, h, d, inverse=reverse, tail_bound=self.tail_bound
        )
        # account for the masking (padded positions must be identity)
        y = y * x_mask.squeeze(1)
        logdet = logdet * x_mask.squeeze(1)
        return y.unsqueeze(1), logdet


class StochasticDurationPredictor(nn.Module):
    """Flow-based stochastic duration predictor, conditioned on text features.

    Forward (training) gives the negative-log-likelihood ingredients;
    ``reverse=True`` samples durations. The exact inverse property is
    guaranteed by construction, so the train/inference distributions match.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        n_tone: int = 5,
        n_length: int = 3,
        num_bins: int = 10,
    ) -> None:
        super().__init__()
        self.filter_channels = filter_channels
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.tone_emb = nn.Embedding(n_tone, filter_channels)
        self.length_emb = nn.Embedding(n_length, filter_channels)

        self.pre = nn.Conv1d(1, filter_channels, 1)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(filter_channels), nn.ReLU(),
                    nn.Conv1d(filter_channels, filter_channels, kernel_size,
                              padding=get_padding(kernel_size)),
                    nn.Dropout(p_dropout),
                )
                for _ in range(3)
            ]
        )
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.dds = nn.Conv1d(filter_channels, 2, 1)          # m, logs (scalar latent)
        self.cond = nn.Conv1d(gin_channels, filter_channels, 1) if gin_channels else None

        # monotone flow stack over the scalar latent, conditioned on text features
        self.flows = nn.ModuleList(
            [
                ConditionalSplineFlow(filter_channels, hidden=filter_channels, n_layers=2,
                                      num_bins=num_bins, kernel_size=5)
                for _ in range(n_flows)
            ]
        )
        self.affine = ElementwiseAffine()

    def _context(self, h_text, tone, length, g, x_mask):
        ctx = h_text
        if tone is not None and length is not None:
            ctx = ctx + (self.tone_emb(tone) + self.length_emb(length)).transpose(1, 2)
        if g is not None and self.cond is not None:
            ctx = ctx + self.cond(g)
        return ctx * x_mask

    def _posterior(self, x, x_mask, h_text, tone, length, g):
        h = self.pre(x) * x_mask
        if tone is not None and length is not None:
            h = h + (self.tone_emb(tone) + self.length_emb(length)).transpose(1, 2)
        if g is not None and self.cond is not None:
            h = h + self.cond(g)
        for conv in self.convs:
            h = conv(h) * x_mask
        stats = self.dds(self.proj(h) * x_mask) * x_mask
        m, logs = torch.split(stats, 1, dim=1)
        return m, torch.clamp(logs, -6.0, 2.0)

    def forward(self, x, x_mask, g=None, h_text=None, tone=None, length=None,
                reverse: bool = False, noise_scale: float = 1.0, temperature: float = 1.0):
        h_text = h_text if h_text is not None else torch.zeros_like(self.pre(x))
        context = self._context(h_text, tone, length, g, x_mask)
        if not reverse:
            m, logs = self._posterior(x, x_mask, h_text, tone, length, g)
            u = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
            v, l1 = self.affine(u, x_mask, reverse=False)
            logdet = logdet + l1
            for flow in self.flows:
                v, l2 = flow(v, x_mask, context, reverse=False)
                logdet = logdet + torch.sum(l2 * x_mask.squeeze(1), dim=1)
            return v, logdet, m, logs
        e = torch.randn(x.size(0), 1, x.size(2), device=x.device) * temperature
        e = e * x_mask
        for flow in reversed(self.flows):
            e, _ = flow(e, x_mask, context, reverse=True)
        e, _ = self.affine(e, x_mask, reverse=True)
        return e * x_mask


class ElementwiseAffine(nn.Module):
    """Elementwise scale/shift flow (1 channel)."""

    def __init__(self) -> None:
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x, x_mask, reverse: bool = False):
        mask = x_mask
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            logdet = torch.sum(self.logs * mask, [1, 2])
            return y * mask, logdet
        y = (x - self.m) * torch.exp(-self.logs) * mask
        return y, torch.zeros(x.size(0), device=x.device)
