"""Core neural blocks for the Hausa VITS extension.

Only the *core* model lives here; the neural vocoder and the discriminators are
in :mod:`hausa_tts.vocoder`, the flow-based duration model in
:mod:`hausa_tts.flows`.

Components that differ from stock VITS (Kim et al. 2021):
``ToneConditionedTextEncoder``, ``PitchPredictor``, ``EnergyPredictor``,
``ProsodyEncoder``/``ProsodyPredictor``, ``SpeakerEncoder`` and
``DurationPredictor`` (tone/length scaled).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    ids = torch.arange(0, max_length, device=length.device, dtype=length.dtype)
    return (ids.unsqueeze(0) < length.unsqueeze(1)).to(torch.bool)


def expand_by_duration(seq: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Repeat each phoneme-level value ``d_i`` times -> frame-level sequence.

    ``seq``: ``[B, T_text]``. ``durations``: ``[B, T_text]`` frames. Rows have
    different totals, so each row is expanded on its own (``repeat_interleave``
    on a 2-D tensor with a 2-D ``repeats`` is not supported) and the batch is
    padded back to the longest row.
    """
    rows = [torch.repeat_interleave(seq[i], durations[i].long()) for i in range(seq.size(0))]
    width = max(r.numel() for r in rows)
    if width == 0:
        return seq.new_zeros(seq.size(0), 0)
    rows = [F.pad(r, (0, width - r.numel())) for r in rows]
    return torch.stack(rows, dim=0)


def onehot_from_attn(attn: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Turn an alignment matrix into frame-level ids of the aligned token."""
    idx = attn.argmax(dim=2)  # [B, T_spec]
    return torch.gather(values, 1, idx)


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# ---------------------------------------------------------------------------
# WaveNet stack + coupling flows (prior / posterior)
# ---------------------------------------------------------------------------


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate**i
            conv = nn.Conv1d(
                hidden_channels,
                hidden_channels * 2,
                kernel_size,
                dilation=dilation,
                padding=get_padding(kernel_size, dilation),
            )
            self.in_layers.append(nn.utils.weight_norm(conv))
            self.res_skip_layers.append(
                nn.utils.weight_norm(nn.Conv1d(hidden_channels, hidden_channels * 2, 1))
            )
        if gin_channels:
            self.cond_layer = nn.utils.weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )
        else:
            self.cond_layer = None

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None):
        output = torch.zeros_like(x)
        gs = None
        if g is not None and self.cond_layer is not None:
            gs = torch.split(self.cond_layer(g), self.hidden_channels * 2, dim=1)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if gs is not None:
                x_in = x_in + gs[i]
            acts = torch.tanh(x_in[:, : self.hidden_channels]) * torch.sigmoid(
                x_in[:, self.hidden_channels :]
            )
            acts = self.drop(acts)
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (res_skip[:, : self.hidden_channels] + x) * x_mask
            output = output + res_skip[:, self.hidden_channels :]
        return output * x_mask


class Flip(nn.Module):
    def forward(self, x, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0), device=x.device)
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ) -> None:
        super().__init__()
        assert channels % 2 == 0
        self.channels = channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels, p_dropout=p_dropout,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        if self.post.bias is not None:
            self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
            logs = torch.clamp(logs, -6.0, 6.0)
        else:
            m, logs = stats, torch.zeros_like(stats)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x, torch.sum(logs, [1, 2])
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], dim=1)


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        mean_only: bool = False,
    ) -> None:
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                    gin_channels=gin_channels, p_dropout=p_dropout, mean_only=mean_only,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=False)
            return x
        for flow in reversed(self.flows):
            x = flow(x, x_mask, g=g, reverse=True)
        return x


# ---------------------------------------------------------------------------
# transformer text encoder
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Transformer attention with a learnable additive relative-position bias.

    The bias is indexed by the *clipped* relative distance, so the model has
    locality control over a phoneme window without the complexity (and cost) of
    the full relative-attention tables used in stock VITS. This matters for
    Hausa because the phonological rules we inject (low-tone raising,
    high-tone spreading) act within a 1-2 syllable window.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        proximal_init: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        if window_size:
            self.rel_bias = nn.Parameter(torch.zeros(window_size * 2 + 1))
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, _, t = x.size()
        q = self.conv_q(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        k = self.conv_k(c).view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        v = self.conv_v(c).view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        scores = torch.matmul(q / math.sqrt(self.k_channels), k.transpose(2, 3))
        if self.window_size:
            i = torch.arange(scores.size(2), device=x.device).unsqueeze(1)
            j = torch.arange(scores.size(3), device=x.device).unsqueeze(0)
            rel = (i - j).clamp(-self.window_size, self.window_size) + self.window_size
            scores = scores + self.rel_bias[rel].unsqueeze(0).unsqueeze(0)
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1).to(torch.bool)
            scores = scores.masked_fill(~mask, -1e4)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(2, 3).contiguous().view(b, self.n_heads * self.k_channels, t)
        return self.conv_o(out)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=get_padding(kernel_size))
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=get_padding(kernel_size))
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.drop(torch.relu(self.conv_1(x * x_mask)))
        return self.drop(self.conv_2(x) * x_mask)


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, window_size=4):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                                   p_dropout=p_dropout, window_size=window_size)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        m = x_mask.squeeze(1)
        attn_mask = m.unsqueeze(2) * m.unsqueeze(1)
        x = x * x_mask
        for attn, n1, ffn, n2 in zip(self.attn_layers, self.norm_layers_1,
                                    self.ffn_layers, self.norm_layers_2):
            y = attn(x, x, attn_mask)
            x = n1(self.drop(y) + x) * x_mask
            y = ffn(x, x_mask)
            x = n2(self.drop(y) + x) * x_mask
        return x * x_mask


class ToneConditionedTextEncoder(nn.Module):
    """Encoder over the phoneme stream, modulated by tone / length / utterance type.

    Three parallel streams (phoneme, tone, vowel length) plus the utterance
    type (statement / yes-no question / wh-question) enter through embeddings
    and a FiLM transform. This is the architectural answer to the fact that
    Boko writing omits tone and vowel length: once the front-end (or a
    diacritizer) supplies them, the model consumes them as *separate* streams
    rather than as extra symbols competing inside one embedding table.
    """

    def __init__(
        self,
        n_vocab: int,
        n_tone: int,
        n_length: int,
        n_utt_type: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float = 0.1,
        window_size: int = 4,
        tone_dim: int = 32,
        length_dim: int = 8,
        utt_dim: int = 8,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.tone_emb = nn.Embedding(n_tone, tone_dim)
        self.length_emb = nn.Embedding(n_length, length_dim)
        self.utt_emb = nn.Embedding(n_utt_type, utt_dim)
        self.film = nn.Linear(tone_dim + length_dim + utt_dim, hidden_channels * 2)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                               kernel_size, p_dropout, window_size=window_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, tone, length, utt_type, x_mask):
        h = (self.emb(x) * math.sqrt(self.emb.embedding_dim)).transpose(1, 2)
        u = self.utt_emb(utt_type).unsqueeze(-1).expand(-1, -1, x.size(1))
        cond = torch.cat(
            [self.tone_emb(tone).transpose(1, 2),
             self.length_emb(length).transpose(1, 2),
             u], dim=1,
        )
        gamma, beta = torch.chunk(self.film(cond.transpose(1, 2)), 2, dim=-1)
        h = h * (1.0 + gamma.transpose(1, 2)) + beta.transpose(1, 2)
        h = self.encoder(h * x_mask, x_mask)
        return self.proj(h) * x_mask


# ---------------------------------------------------------------------------
# posterior encoder
# ---------------------------------------------------------------------------


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None):
        x = self.enc(self.pre(x) * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        logs = torch.clamp(logs, -6.0, 2.0)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


# ---------------------------------------------------------------------------
# prosody / pitch / energy
# ---------------------------------------------------------------------------


class SinusoidalPositionEncoding(nn.Module):
    """Continuous pitch representation: embeddings of a scalar log-F0 track.

    A continuous (rather than quantised) pitch representation preserves the
    micro-prosody that carries Hausa downdrift and question intonation
    (Hayes et al. 2024 on differentiable DSP; Nercessian 2022).
    """

    def __init__(self, channels: int, n_freqs: int = 20) -> None:
        super().__init__()
        self.proj = nn.Linear(n_freqs * 2 + 1, channels)
        self.register_buffer("freqs", torch.linspace(1.0, 8.0, n_freqs), persistent=False)

    def forward(self, f0_log: torch.Tensor) -> torch.Tensor:
        x = f0_log.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        feats = torch.cat([torch.sin(x), torch.cos(x), f0_log.unsqueeze(-1)], dim=-1)
        return self.proj(feats)


class ProsodyEncoder(nn.Module):
    """Variational encoder over the 8 interpretable utterance prosody statistics."""

    n_stats = 8

    def __init__(self, out_channels: int = 64, hidden: int = 128, in_stats: int = n_stats) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_stats, hidden), nn.ReLU(), nn.Linear(hidden, out_channels * 2))

    def forward(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m, logs = torch.chunk(self.net(stats), 2, dim=-1)
        logs = torch.clamp(logs, -6.0, 2.0)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m


class ProsodyPredictor(nn.Module):
    """Text -> utterance prosody statistics (used when no reference audio exists)."""

    def __init__(self, in_channels: int, hidden: int = 128, out_stats: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_stats),
        )

    def forward(self, h_text, x_mask):
        pooled = (h_text * x_mask).sum(-1) / x_mask.sum(-1).clamp_min(1.0)
        return self.net(pooled)


class PitchPredictor(nn.Module):
    """[HAUSA] Frame-level pitch predictor.

    Predicts a *residual* on top of the rule-based Hausa F0 targets, so the
    model starts from a linguistically correct contour (declination, downdrift,
    lexical tone levels, question intonation) and only has to learn the
    speaker- and segment-specific detail that the rules cannot express. Tone and
    vowel length are injected as one-hot streams, so the same phoneme can carry a
    different pitch under a different lexical tone without the text encoder
    having to disentangle the two.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3, n_layers=2,
                 n_tone=4, n_length=3, gin_channels=0, **kwargs):
        super().__init__()
        self.n_tone = n_tone
        self.n_length = n_length
        cond = 1 + n_tone + n_length          # f0 hint + tone one-hot + length one-hot
        self.pre = nn.Conv1d(in_channels + cond, hidden_channels, kernel_size,
                             padding=kernel_size // 2)
        self.layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                      padding=(kernel_size // 2) * (2 ** i), dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden_channels) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        self.gin_cond = nn.Conv1d(gin_channels, hidden_channels, 1) if gin_channels else None

    def forward(self, h, tone_frames, length_frames, f0_hint, x_mask, g=None):
        tone_oh = F.one_hot(tone_frames.long().clamp(0, self.n_tone - 1), self.n_tone)
        len_oh = F.one_hot(length_frames.long().clamp(0, self.n_length - 1), self.n_length)
        streams = [h, tone_oh.transpose(1, 2).float(), len_oh.transpose(1, 2).float(),
                   f0_hint.unsqueeze(1)]
        t_ref = int(h.size(2))
        streams = [st if int(st.size(2)) == t_ref else
                   F.interpolate(st.float(), size=t_ref, mode="nearest") for st in streams]
        m = x_mask if int(x_mask.size(2)) == t_ref else F.interpolate(
            x_mask.float(), size=t_ref, mode="nearest")
        x = torch.cat(streams, dim=1) * m
        x = self.pre(x)
        if g is not None and self.gin_cond is not None:
            x = x + self.gin_cond(g)
        for conv, norm in zip(self.layers, self.norms):
            x = x + torch.relu(norm(conv(F.leaky_relu(x, 0.1))))
        return self.proj(F.relu(x)) * x_mask


class EnergyPredictor(nn.Module):
    """[HAUSA] Frame-level energy (intensity) predictor.

    Energy is the prosodic correlate of stress and of the pitch-accent
    prominence that Hausa marks alongside tone; predicting it explicitly keeps
    the F0 stream from having to absorb loudness information.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3, n_layers=2,
                 gin_channels=0, **kwargs):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size,
                             padding=kernel_size // 2)
        self.layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                      padding=(kernel_size // 2) * (2 ** i), dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden_channels) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        self.gin_cond = nn.Conv1d(gin_channels, hidden_channels, 1) if gin_channels else None

    def forward(self, h, x_mask, g=None):
        x = self.pre(h * x_mask)
        if g is not None and self.gin_cond is not None:
            x = x + self.gin_cond(g)
        for conv, norm in zip(self.layers, self.norms):
            x = x + torch.relu(norm(conv(F.leaky_relu(x, 0.1))))
        return self.proj(F.relu(x)) * x_mask


class ToneFromF0Classifier(nn.Module):
    """Predict the lexical tone of each vowel *from the synthesised F0 contour*.

    This gives a differentiable (and at evaluation time reportable) tonal
    fidelity signal: mean level, slope, range and duration of the contour over
    each vowel are mapped to H / L / F / R. Because it reads the *contour*, a
    tone error cannot be hidden by a fluent-sounding waveform.
    """

    n_feats = 5
    n_classes = 5

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.n_feats, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


# ---------------------------------------------------------------------------
# duration model (deterministic; the stochastic one lives in flows.py)
# ---------------------------------------------------------------------------


class DurationPredictor(nn.Module):
    """Log-domain duration predictor with tone- and length-aware scaling.

    A learnable per-(tone, length) bias is added in the log domain, which
    brings in the well-attested fact that Hausa long vowels are roughly
    1.7-2.0x short ones and that low tones are often longer than high tones
    (Newman & Van Heuven 1981; Newman 1997). Unlike a generic duration
    predictor, the model cannot collapse the length contrast because the bias
    is a separate parameter per (tone, length) pair.
    """

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout,
                 gin_channels=0, n_tone=5, n_length=3, length_scaling=True):
        super().__init__()
        self.length_scaling = length_scaling
        self.tone_emb = nn.Embedding(n_tone, filter_channels)
        self.length_emb = nn.Embedding(n_length, filter_channels)
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(filter_channels), nn.ReLU(),
                    nn.Conv1d(filter_channels, filter_channels, kernel_size,
                              padding=get_padding(kernel_size)),
                    nn.Dropout(p_dropout),
                )
                for _ in range(2)
            ]
        )
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.cond = nn.Conv1d(gin_channels, filter_channels, 1) if gin_channels else None
        if length_scaling:
            self.log_scale = nn.Parameter(torch.zeros(n_tone, n_length))

    def forward(self, x, x_mask, g=None, tone=None, length=None):
        h = self.pre(x) * x_mask
        if tone is not None and length is not None:
            h = h + (self.tone_emb(tone) + self.length_emb(length)).transpose(1, 2)
        if g is not None and self.cond is not None:
            h = h + self.cond(g)
        for conv in self.convs:
            h = conv(h) * x_mask
        out = self.proj(h) * x_mask
        if self.length_scaling and tone is not None and length is not None:
            out = out + self.log_scale[tone, length].unsqueeze(-1).transpose(1, 2)
        return out * x_mask


# ---------------------------------------------------------------------------
# speaker encoder
# ---------------------------------------------------------------------------


class SpeakerEncoder(nn.Module):
    """Compact ECAPA-TDNN speaker encoder (Desplanques et al. 2020).

    Enables zero-shot multi-speaker synthesis in the YourTTS sense
    (Casanova et al. 2022): at inference a few seconds of a *new* speaker's
    audio are embedded into ``g`` and condition the whole model.
    """

    def __init__(self, n_mels: int = 80, channels: int = 128, emb_dim: int = 192,
                 res_blocks: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels if i == 0 else channels * 3, channels * 3, 3, padding=1),
                    nn.BatchNorm1d(channels * 3), nn.ReLU(),
                )
                for i in range(res_blocks)
            ]
        )
        self.mfa = nn.Conv1d(channels + channels * 3 * res_blocks, channels * 3, 1)
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, 128, 1), nn.Tanh(), nn.Conv1d(128, channels * 3, 1)
        )
        self.fc = nn.Linear(channels * 6, emb_dim)

    def forward(self, mel: torch.Tensor, mel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mel.dim() == 4:            # [B, 1, M, T] -> [B, M, T]
            mel = mel[:, 0]
        if mel.dim() == 2:            # [M, T] -> [1, M, T]
            mel = mel.unsqueeze(0)
        if mel_mask is not None and mel_mask.dim() == 3:
            mel_mask = mel_mask.squeeze(1)
        if mel_mask is not None:
            mel = mel * mel_mask.unsqueeze(1)
        x = F.relu(self.conv1(mel))
        feats = [x]
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        x = self.mfa(torch.cat(feats, dim=1))
        attn = torch.softmax(self.attention(x), dim=2)
        mu = (x * attn).sum(dim=2)
        var = ((x**2) * attn).sum(dim=2) - mu**2
        sd = torch.sqrt(var.clamp_min(1e-8))
        return self.fc(torch.cat([mu, sd], dim=1))
