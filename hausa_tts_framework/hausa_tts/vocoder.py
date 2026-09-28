"""Neural vocoder and discriminators.

``F0ConditionedGenerator`` is a HiFi-GAN generator (Kong et al. 2020) with an
explicit **pitch-hint injection** at every upsampling stage. The hint is the
continuous log-F0 contour produced by the Hausa prosody model (or by the pitch
predictor). This is the source-filter-like design that gives the system direct
control over the F0 that actually ends up in the waveform -- essential in a
tonal language where a wrong F0 contour changes the word.

``ToneContourDiscriminator`` is the Hausa-specific adversarial critic: it
judges the *pitch contour together with its tone/length annotation*, so a
tonally wrong realisation is punished even when it sounds like fluent speech.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .modules import SinusoidalPositionEncoding, get_padding

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# HiFi-GAN blocks
# ---------------------------------------------------------------------------


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                              padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                              padding=get_padding(kernel_size, 1))
                )
                for _ in dilation
            ]
        )

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = F.leaky_relu(c1(xt), LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x * x_mask if x_mask is not None else x

    def remove_weight_norm(self):
        for c in self.convs1:
            torch.nn.utils.remove_weight_norm(c)
        for c in self.convs2:
            torch.nn.utils.remove_weight_norm(c)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                              padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(c(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)
            x = xt + x
        return x * x_mask if x_mask is not None else x

    def remove_weight_norm(self):
        for c in self.convs:
            torch.nn.utils.remove_weight_norm(c)


class F0ConditionedGenerator(nn.Module):
    """HiFi-GAN generator with pitch-hint conditioning at every scale."""

    def __init__(
        self,
        in_channels: int = 192,
        upsample_initial_channel: int = 512,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock: str = "1",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        gin_channels: int = 0,
        use_f0_hint: bool = True,
        f0_hint_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_f0_hint = use_f0_hint
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.hint_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(upsample_initial_channel // (2**i),
                                       upsample_initial_channel // (2 ** (i + 1)),
                                       k, u, padding=(k - u) // 2)
                )
            )
            if use_f0_hint:
                self.hint_convs.append(
                    nn.Conv1d(f0_hint_dim, upsample_initial_channel // (2 ** (i + 1)), 1)
                )
        if use_f0_hint:
            self.f0_enc = SinusoidalPositionEncoding(f0_hint_dim)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(upsample_initial_channel // (2 ** len(self.ups)), 1, 7, 1, padding=3)
        )
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels else None

    def forward(self, x, g=None, f0_hint: Optional[torch.Tensor] = None):
        x = self.conv_pre(x)
        if self.cond is not None and g is not None:
            x = x + self.cond(g)
        if self.cond is not None and g is None:
            raise ValueError("gin_channels is set but no speaker embedding `g` was passed")
        hint_feat = None
        if self.use_f0_hint:
            if f0_hint is None:
                f0_hint = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
            hint_feat = self.f0_enc(f0_hint.float()).transpose(1, 2)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            if hint_feat is not None:
                x = x + self.hint_convs[i](
                    F.interpolate(hint_feat, size=x.size(2), mode="linear", align_corners=False)
                )
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def remove_weight_norm(self):
        torch.nn.utils.remove_weight_norm(self.conv_pre)
        for up in self.ups:
            torch.nn.utils.remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        torch.nn.utils.remove_weight_norm(self.conv_post)


# ---------------------------------------------------------------------------
# discriminators
# ---------------------------------------------------------------------------


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm = torch.nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            d_r, f_r = d(y)
            d_g, f_g = d(y_hat)
            y_d_rs.append(d_r); fmap_rs.append(f_r)
            y_d_gs.append(d_g); fmap_gs.append(f_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = torch.nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            d_r, f_r = d(y)
            d_g, f_g = d(y_hat)
            y_d_rs.append(d_r); fmap_rs.append(f_r)
            y_d_gs.append(d_g); fmap_gs.append(f_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ToneContourDiscriminator(nn.Module):
    """Tonal adversarial critic over the F0 contour + tone/length annotation."""

    def __init__(self, n_tone: int = 5, n_length: int = 3, hidden: int = 128) -> None:
        super().__init__()
        self.tone_emb = nn.Embedding(n_tone, 16)
        self.length_emb = nn.Embedding(n_length, 8)
        in_ch = 1 + 16 + 8
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, 1, padding=2), nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden, hidden * 2, 5, 2, padding=2), nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden * 2, hidden * 2, 5, 2, padding=2), nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden * 2, hidden * 4, 5, 2, padding=2), nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden * 4, hidden * 4, 3, 1, padding=1),
        )
        self.head = nn.Conv1d(hidden * 4, 1, 1)

    def forward(self, f0: torch.Tensor, tone_frames: torch.Tensor, length_frames: torch.Tensor):
        cond = torch.cat(
            [self.tone_emb(tone_frames).transpose(1, 2),
             self.length_emb(length_frames).transpose(1, 2)], dim=1
        )
        cond = F.interpolate(cond, size=f0.size(2), mode="nearest")
        return self.head(self.net(torch.cat([f0, cond], dim=1)))


def get_padding_(kernel_size: int, dilation: int = 1) -> int:  # pragma: no cover
    return get_padding(kernel_size, dilation)
