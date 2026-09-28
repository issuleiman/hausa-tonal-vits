"""The Hausa VITS extension: single-speaker and multi-speaker end-to-end TTS.

Pipeline (all trained jointly, as in VITS):

    text  --(tone/length separate streams)-->  tone-conditioned text encoder
          --> prosody predictor (utterance statistics)
          --> duration model (deterministic or stochastic, tone/length scaled)
          --> prior flow  <-->  posterior encoder (mel spectrogram)
          --> pitch predictor (residual on the rule-based Hausa F0 targets)
          --> energy predictor
          --> HiFi-GAN decoder conditioned on the continuous F0 hint
          --> waveform

Differences from VITS / VITS2 are marked ``[HAUSA]`` in the code so the
contribution of this work is auditable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import alignment as mas
from .flows import StochasticDurationPredictor
from .modules import (
    DurationPredictor,
    EnergyPredictor,
    PitchPredictor,
    PosteriorEncoder,
    ProsodyEncoder,
    ProsodyPredictor,
    ResidualCouplingBlock,
    SpeakerEncoder,
    ToneConditionedTextEncoder,
    ToneFromF0Classifier,
    expand_by_duration,
    init_weights,
    onehot_from_attn,
    sequence_mask,
)
from .vocoder import F0ConditionedGenerator


# a Hausa phone is never longer than ~1 s; anything beyond this (possible with
# an untrained stochastic duration predictor) is a numerical artefact, not speech.
MAX_PHONE_FRAMES = 60


class HausaVITS(nn.Module):
    """End-to-end tonal Hausa TTS model built on the VITS architecture."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        n_vocab = cfg.n_vocab
        attn_channels = cfg.inter_channels * cfg.attn_hidden_scale
        gin = 0 if cfg.speaker_source == "none" else cfg.gin_channels

        # ---- text encoder: [HAUSA] three streams + utterance type -----------
        self.enc_p = ToneConditionedTextEncoder(
            n_vocab=n_vocab,
            n_tone=cfg.n_tone,
            n_length=cfg.n_length,
            n_utt_type=cfg.n_utt_type,
            out_channels=attn_channels,
            hidden_channels=cfg.hidden_channels,
            filter_channels=cfg.filter_channels,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            kernel_size=cfg.kernel_size,
            p_dropout=cfg.p_dropout,
            window_size=cfg.window_size,
        )
        self.proj = nn.Conv1d(attn_channels, cfg.inter_channels * 2, 1)
        # [HAUSA] frame-level prosody heads operate in the latent space, so the
        # duration-expanded text features are projected once and reused.
        self.h_proj = nn.Linear(attn_channels, cfg.inter_channels)

        # ---- posterior + prior flows ---------------------------------------
        self.enc_q = PosteriorEncoder(
            in_channels=cfg.spec_channels,
            out_channels=cfg.inter_channels,
            hidden_channels=cfg.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin,
        )
        self.flow = ResidualCouplingBlock(
            channels=cfg.inter_channels,
            hidden_channels=cfg.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=cfg.n_flows,
            gin_channels=gin,
        )

        # ---- [HAUSA] duration model ----------------------------------------
        self.duration_type = cfg.duration_type
        if cfg.duration_type == "stochastic":
            self.dp = StochasticDurationPredictor(
                in_channels=attn_channels,
                filter_channels=cfg.attn_hidden_scale * cfg.inter_channels,
                kernel_size=3,
                p_dropout=cfg.p_dropout,
                n_flows=cfg.n_flows_dur,
                gin_channels=gin,
                n_tone=cfg.n_tone,
                n_length=cfg.n_length,
            )
        else:
            self.dp = DurationPredictor(
                in_channels=attn_channels,
                filter_channels=cfg.hidden_channels,
                kernel_size=3,
                p_dropout=cfg.p_dropout,
                gin_channels=gin,
                n_tone=cfg.n_tone,
                n_length=cfg.n_length,
                length_scaling=cfg.length_scaled_duration,
            )

        # ---- [HAUSA] prosody / pitch / energy ------------------------------
        self.prosody_enc = ProsodyEncoder(out_channels=cfg.prosody_channels)
        self.prosody_pred = ProsodyPredictor(
            in_channels=attn_channels, hidden=cfg.hidden_channels, out_stats=ProsodyEncoder.n_stats
        )
        self.prosody_proj = nn.Linear(cfg.prosody_channels, cfg.prosody_emb)
        self.pitch_pred = PitchPredictor(
            in_channels=cfg.inter_channels,
            hidden_channels=cfg.pitch_hidden,
            kernel_size=3,
            n_layers=cfg.pitch_layers,
            n_tone=cfg.n_tone,
            n_length=cfg.n_length,
            gin_channels=gin,
        )
        self.energy_pred = EnergyPredictor(
            in_channels=cfg.inter_channels, hidden_channels=cfg.hidden_channels,
            kernel_size=3, n_layers=2, gin_channels=gin,
        )
        self.tone_cls = ToneFromF0Classifier()

        # ---- decoder --------------------------------------------------------
        self.dec = F0ConditionedGenerator(
            in_channels=cfg.inter_channels + 1 + cfg.prosody_emb
            + (cfg.prosody_channels if cfg.use_prosody_latent else 0),
            upsample_initial_channel=cfg.upsample_initial_channel,
            upsample_rates=tuple(cfg.upsample_rates),
            upsample_kernel_sizes=tuple(cfg.upsample_kernel_sizes),
            resblock=cfg.resblock,
            resblock_kernel_sizes=tuple(cfg.resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(tuple(d) for d in cfg.resblock_dilation_sizes),
            gin_channels=gin,
            use_f0_hint=cfg.use_f0_hint,
            f0_hint_dim=cfg.f0_hint_dim,
        )

        # ---- speaker conditioning ------------------------------------------
        if cfg.speaker_source in ("id", "both"):
            self.emb_g = nn.Embedding(cfg.n_speakers, cfg.gin_channels)
        else:
            self.emb_g = None
        if cfg.speaker_source in ("reference", "both"):
            self.spk_enc = SpeakerEncoder(
                n_mels=cfg.n_mels, channels=cfg.speaker_enc_channels, emb_dim=cfg.gin_channels
            )
        else:
            self.spk_enc = None

        self.dec.apply(init_weights)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_parts(parts):
        """Trim/expand every decoder input to one common frame length.

        The latent stream, the predicted energy and the prosody streams are
        produced by different branches; a branch that collapsed to a single
        frame (e.g. a globally broadcasted latent) is expanded so that the
        concatenation can never silently mis-align the acoustic features.
        """
        t_max = max(int(p.size(2)) for p in parts)
        out = []
        for p in parts:
            if int(p.size(2)) == 1 and t_max > 1:
                p = p.expand(-1, -1, t_max)
            elif int(p.size(2)) != t_max:
                p = F.interpolate(p, size=t_max, mode="nearest")
            out.append(p[:, :, :t_max])
        return out

    def _speaker_embedding(self, sid, reference_mel, reference_mel_mask) -> Optional[torch.Tensor]:
        g = None
        if self.emb_g is not None and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        if self.spk_enc is not None and reference_mel is not None:
            g_ref = self.spk_enc(reference_mel, reference_mel_mask).unsqueeze(-1)
            g = g_ref if g is None else g + g_ref
        if self.cfg.speaker_source == "none":
            return None
        if g is None:
            b = reference_mel.size(0) if reference_mel is not None else sid.size(0)
            g = torch.zeros(b, self.cfg.gin_channels, 1, device=self.proj.weight.device)
        return g

    def _prior_stats(self, h_text, x_mask):
        stats = self.proj(h_text) * x_mask
        m_p, logs_p = torch.chunk(stats, 2, dim=1)
        return m_p, torch.clamp(logs_p, -6.0, 2.0)

    def _alignment(self, z_p, m_p, logs_p, x_mask, spec_mask, vowel_mask, pause_mask):
        """Monotonic alignment search between the text prior and the latent.

        Returns a *spec-major* path ``[B, 1, T_spec, T_text]`` so that
        ``matmul(attn, m_p.transpose(1, 2))`` expands the text-level prior onto
        the frame axis (the VITS convention).
        """
        import math

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4      # [B, T_spec, T_text]
            if self.cfg.align_prior:
                # [HAUSA] vowels attract frames, pauses repel them: a linguistically
                # motivated prior on the alignment cost.
                neg_cent = (neg_cent
                            + 0.05 * vowel_mask.unsqueeze(1)
                            - 0.05 * pause_mask.unsqueeze(1))
            attn_mask = x_mask.squeeze(1).unsqueeze(2) * spec_mask.squeeze(1).unsqueeze(1)
            path = mas.maximum_path(neg_cent.transpose(1, 2), attn_mask)  # [B, T_text, T_spec]
            attn = path.transpose(1, 2).unsqueeze(1) * x_mask.unsqueeze(2)
        return attn

    def _alignment_from_durations(self, duration_frames, x_mask, spec_mask):
        """Deterministic path from known phone durations -> ``[B, 1, T_spec, T_text]``."""
        path = mas.generate_path(duration_frames, spec_mask.unsqueeze(1))   # [B, T_text, T_spec]
        return path.transpose(1, 2).unsqueeze(1) * x_mask.unsqueeze(2)

    def _prosody(
        self, h_text, x_mask, prosody_stats, g, device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (prosody embedding for the decoder, sampled latent, predicted stats)."""
        stats_pred = self.prosody_pred(h_text, x_mask)
        stats = prosody_stats if prosody_stats is not None else stats_pred
        z_pros, m_pros = self.prosody_enc(stats)
        emb = self.prosody_proj(z_pros)
        return emb, z_pros, stats_pred

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        length: torch.Tensor,
        utt_type: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        f0_hint_rule: torch.Tensor,
        f0_target: torch.Tensor,
        energy_target: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        reference_mel_mask: Optional[torch.Tensor] = None,
        prosody_stats: Optional[torch.Tensor] = None,
        duration_frames: Optional[torch.Tensor] = None,
        output_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x_mask = sequence_mask(x_lengths, x.size(1)).unsqueeze(1)
        spec_mask = sequence_mask(spec_lengths, spec.size(2)).unsqueeze(1)
        # `y` is optional at training time: the adversarial part of the loss
        # needs the raw waveform, but the reconstruction part only needs mel.
        n_wav = y.size(2) if y is not None else int(y_lengths.max().item())
        y_mask = sequence_mask(y_lengths, n_wav).unsqueeze(1)

        h_text = self.enc_p(x, tone, length, utt_type, x_mask)
        m_p, logs_p = self._prior_stats(h_text, x_mask)
        g = self._speaker_embedding(sid, reference_mel, reference_mel_mask)

        z, m_q, logs_q = self.enc_q(spec, spec_mask, g=g)
        z_p = self.flow(z, spec_mask, g=g)

        vowel_mask = torch.isin(x, torch.tensor(self.cfg.vowel_ids, device=x.device)).float()
        pause_mask = torch.isin(x, torch.tensor(self.cfg.silence_ids, device=x.device)).float()

        if duration_frames is not None:
            attn = self._alignment_from_durations(duration_frames, x_mask, spec_mask)
        else:
            attn = self._alignment(z_p, m_p, logs_p, x_mask, spec_mask, vowel_mask, pause_mask)

        attn_sq = attn.squeeze(1)
        x_dur = attn_sq.sum(1)                        # [B, T_text] frames per phoneme
        x_dur_log = torch.log(x_dur.clamp_min(1e-4)) * x_mask.squeeze(1)

        m_p_e = torch.matmul(attn_sq, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_e = torch.matmul(attn_sq, logs_p.transpose(1, 2)).transpose(1, 2)
        z_p_e = z_p  # already frame-level; only the text-level prior is expanded
        h_e = torch.matmul(attn_sq, h_text.transpose(1, 2)).transpose(1, 2)
        h_e_pros = self.h_proj(h_e.transpose(1, 2)).transpose(1, 2) * spec_mask

        # frame-level tone / length (from the alignment)
        tone_frames = onehot_from_attn(attn_sq, tone)
        length_frames = onehot_from_attn(attn_sq, length)

        # ---- prosody --------------------------------------------------------
        prosody_emb, z_pros, stats_pred = self._prosody(h_text, x_mask, prosody_stats, g, x.device)
        prosody_frames = prosody_emb.unsqueeze(-1).expand(-1, -1, spec.size(2)) * spec_mask

        # ---- pitch (residual on the rule-based target) -----------------------
        f0_res = self.pitch_pred(h_e_pros, tone_frames, length_frames, f0_hint_rule, spec_mask, g=g)
        f0_pred = f0_hint_rule + f0_res.squeeze(1)
        f0_pred = f0_pred * spec_mask.squeeze(1)

        # ---- energy ---------------------------------------------------------
        energy_pred = self.energy_pred(h_e_pros, spec_mask, g=g) * spec_mask

        # ---- duration -------------------------------------------------------
        if self.duration_type == "stochastic":
            v, logdet, m_dur, logs_dur = self.dp(
                x_dur_log.unsqueeze(1), x_mask, g=g, h_text=h_text,
                tone=tone, length=length, reverse=False,
            )
            dur_pred_log = self.dp(
                x_dur_log.unsqueeze(1), x_mask, g=g, h_text=h_text,
                tone=tone, length=length, reverse=True,
            ).squeeze(1)
            dur_pack = {"v": v, "logdet": logdet, "m_dur": m_dur, "logs_dur": logs_dur}
        else:
            dur_pred_log = self.dp(h_text, x_mask, g=g, tone=tone, length=length).squeeze(1)
            dur_pack = {}

        # ---- prior-space decoding -------------------------------------------
        z_dec = self.flow(z_p_e, spec_mask, g=g, reverse=True)
        dec_parts = [z_dec, energy_pred, prosody_frames]
        if self.cfg.use_prosody_latent:
            dec_parts.append(
                z_pros.unsqueeze(-1).expand(-1, -1, spec.size(2)) * spec_mask
            )
        o = self.dec(torch.cat(self._fit_parts(dec_parts), dim=1), g=g, f0_hint=f0_pred)

        out = {
            "o": o,
            "y": y,
            "z_dec": z_dec,
            "m_q": m_q,
            "logs_q": logs_q,
            "m_p": m_p_e,
            "logs_p": logs_p_e,
            "spec_mask": spec_mask,
            "y_mask": y_mask,
            "x_mask": x_mask,
            "x_dur": x_dur,
            "x_dur_log": x_dur_log,
            "dur_pred_log": dur_pred_log,
            # exported text-major for the tone-segment builder: [B, T_text, T_spec]
            "attn": attn_sq.transpose(1, 2),
            "f0_pred": f0_pred,
            "f0_target": f0_target,
            "f0_mask": spec_mask.squeeze(1),
            "energy_pred": energy_pred.squeeze(1),
            "energy_target": energy_target,
            "prosody_stats_pred": stats_pred,
            "prosody_stats_target": prosody_stats,
            "tone_frames": tone_frames,
            "length_frames": length_frames,
            "tone": tone,
            "length": length,
            "vowel_mask": vowel_mask,
            "h_text": h_text,
            "h_e": h_e,
            "z_p_e": z_p_e,
            "z_pros": z_pros,
            "g": g,
        }
        out.update(dur_pack)
        if output_attn:
            length_scale = self.cfg.upsample_rates[0]
            for _ in self.cfg.upsample_rates[1:]:
                length_scale *= _
            out["attn_upsampled"] = F.interpolate(
                attn_sq.transpose(1, 2).unsqueeze(1), scale_factor=length_scale * 1, mode="nearest"
            ).squeeze(1)
        return out

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        tone: torch.Tensor,
        length: torch.Tensor,
        utt_type: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        reference_mel_mask: Optional[torch.Tensor] = None,
        prosody_stats: Optional[torch.Tensor] = None,
        duration_override: Optional[torch.Tensor] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        f0_scale: float = 1.0,
        energy_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesise a waveform. Returns ``(audio, attn, f0, durations)``."""
        x_mask = sequence_mask(
            torch.tensor([x.size(1)] * x.size(0), device=x.device), x.size(1)
        ).unsqueeze(1)
        h_text = self.enc_p(x, tone, length, utt_type, x_mask)
        m_p, logs_p = self._prior_stats(h_text, x_mask)
        g = self._speaker_embedding(sid, reference_mel, reference_mel_mask)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # ---- durations ------------------------------------------------------
        if duration_override is not None:
            w = duration_override.float()
        elif self.duration_type == "stochastic":
            e = self.dp(
                # the duration latent is a single scalar channel, not a copy of h_text
                torch.zeros(x.size(0), 1, 1, device=x.device), x_mask, g=g, h_text=h_text,
                tone=tone, length=length, reverse=True, noise_scale=noise_scale_dur,
            )
            w = torch.exp(e.squeeze(1)) * x_mask.squeeze(1)
        else:
            w = torch.exp(
                self.dp(h_text, x_mask, g=g, tone=tone, length=length).squeeze(1)
            ) * x_mask.squeeze(1)
        w = w * length_scale
        w_ceil = (torch.ceil(w.clamp(min=1.0, max=float(MAX_PHONE_FRAMES))).long()
                  * x_mask.squeeze(1).long())

        spec_mask = sequence_mask(w_ceil.sum(-1), int(w_ceil.sum(-1).max())).unsqueeze(1)
        attn = self._alignment_from_durations(w_ceil, x_mask, spec_mask)

        m_p_e = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_e = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p_e = torch.matmul(attn.squeeze(1), z_p.transpose(1, 2)).transpose(1, 2)
        # the duration grid can differ from the matmul output by a frame: the mask
        # must follow the latent, otherwise the flow stack sees mis-matched streams.
        t_lat = int(z_p_e.size(2))
        if int(spec_mask.size(2)) != t_lat:
            spec_mask = (spec_mask if int(spec_mask.size(2)) > t_lat else
                         F.pad(spec_mask, (0, t_lat - int(spec_mask.size(2)))))[:, :, :t_lat]
        h_e = torch.matmul(attn.squeeze(1), h_text.transpose(1, 2)).transpose(1, 2)
        h_e_pros = self.h_proj(h_e.transpose(1, 2)).transpose(1, 2) * spec_mask

        tone_frames = expand_by_duration(tone, w_ceil)
        length_frames = expand_by_duration(length, w_ceil)
        if tone_frames.size(1) < spec_mask.size(2):
            pad_n = spec_mask.size(2) - tone_frames.size(1)
            tone_frames = F.pad(tone_frames, (0, pad_n))
            length_frames = F.pad(length_frames, (0, pad_n))

        # rule-based prosodic F0 targets expanded by the predicted durations
        prosody_emb, z_pros, stats_pred = self._prosody(h_text, x_mask, prosody_stats, g, x.device)
        prosody_frames = prosody_emb.unsqueeze(-1).expand(-1, -1, spec_mask.size(2)) * spec_mask
        pitch_level = torch.clamp(stats_pred[:, 0:1], 3.0, 6.5)
        f0_rule = self._rule_based_f0(tone_frames, length_frames, utt_type, spec_mask) + pitch_level
        f0_res = self.pitch_pred(h_e_pros, tone_frames, length_frames, f0_rule, spec_mask, g=g)
        f0_pred = torch.nan_to_num(
            (f0_rule + f0_scale * f0_res.squeeze(1)) * spec_mask.squeeze(1),
            nan=0.0, posinf=8.0, neginf=-8.0)

        energy_pred = torch.nan_to_num(
            self.energy_pred(h_e_pros, spec_mask, g=g) * spec_mask * energy_scale,
            nan=0.0, posinf=1e3, neginf=-1e3)

        z_dec = self.flow(z_p_e, spec_mask, g=g, reverse=True)
        dec_parts = [z_dec, energy_pred, prosody_frames]
        if self.cfg.use_prosody_latent:
            dec_parts.append(z_pros.unsqueeze(-1).expand(-1, -1, spec_mask.size(2)) * spec_mask)
        audio = self.dec(torch.cat(self._fit_parts(dec_parts), dim=1), g=g, f0_hint=f0_pred)
        # HiFi-GAN upsampling fixes the sample rate only up to a frame boundary;
        # force the exact duration implied by the predicted phone durations so the
        # emitted waveform length is deterministic and auditable.
        want = int(w_ceil.sum(-1).max()) * self.cfg.hop_length
        if int(audio.size(2)) > want:
            audio = audio[:, :, :want]
        elif int(audio.size(2)) < want:
            audio = F.pad(audio, (0, want - int(audio.size(2))))
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio, attn.squeeze(1), f0_pred, w_ceil

    def _rule_based_f0(self, tone_frames, length_frames, utt_type, spec_mask) -> torch.Tensor:
        """Differentiable-ish rule-based F0 target in the log domain.

        A per-tone base level plus declination over the utterance, computed on
        the fly in torch so the model can be trained end-to-end without a
        second data pass. Values are in log-Hz-like units and are later shifted
        by speaker statistics, and they encode exactly the phonology of
        :mod:`hausa_tts.text`: H = 0, L = -0.34, Raised-L = -0.16,
        Falling = ramped, plus declination and a yes/no final rise.
        """
        levels = torch.tensor([0.0, self.cfg.h_level, self.cfg.l_level, self.cfg.f_level, self.cfg.r_level], device=tone_frames.device)
        base = levels[tone_frames]                              # [B, T]
        t = torch.arange(base.size(1), device=base.device).float()
        t = t / t[-1:].clamp_min(1.0)
        base = base - self.cfg.declination * t                                   # declination
        is_yesno = (utt_type == 1).float().unsqueeze(-1)
        base = base + is_yesno * self.cfg.question_rise * torch.sigmoid(8 * (t - 0.85))  # clause-final rise
        is_wh = (utt_type == 2).float().unsqueeze(-1)
        base = base + is_wh * self.cfg.wh_boost
        return base * spec_mask.squeeze(1)
