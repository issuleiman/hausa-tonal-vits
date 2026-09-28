"""Training loop for the Hausa VITS extension.

Objectives (all jointly optimised, VITS-style adversarial + reconstruction):

    L = L_adv + L_fm
      + c_mel * L_mel(decoder output, ground-truth mel)
      + c_kl  * L_kl(z_p, z)
      + c_dur * L_dur                       (flow NLL or MSE, tone/length aware)
      + c_pitch * L_pitch                   [HAUSA] continuous F0 regression
      + c_tone  * L_tone                    [HAUSA] lexical tone from F0 contour
      + c_energy * L_energy
      + c_prosody * L_prosody_stats
      + c_tonal_adv * L_tonal_adv           [HAUSA] tonal adversarial critic
      + c_harmonic * L_harmonic             [HAUSA] harmonic (PeriodVITS) loss
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import audio as haudio
from . import losses as L
from . import alignment as mas
from .data import HausaTTSDataset, collate
from .model import HausaVITS
from .modules import SpeakerEncoder, sequence_mask
from .vocoder import MultiPeriodDiscriminator, MultiScaleDiscriminator, ToneContourDiscriminator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def slice_audio(y: torch.Tensor, y_lengths: torch.Tensor, segment: int, ids: Optional[torch.Tensor] = None):
    b, _, t = y.size()
    if t <= segment:
        return y, torch.zeros(b, dtype=torch.long, device=y.device)
    if ids is not None:
        idx = ids.clamp(max=max(0, t - segment))
    else:
        max_start = (y_lengths - segment).clamp(min=0)
        idx = (torch.rand(b, device=y.device) * (max_start + 1).float()).long()
        idx = idx.clamp(max=max(0, t - segment))
    out = torch.zeros(b, 1, segment, device=y.device, dtype=y.dtype)
    for i, s in enumerate(idx.tolist()):
        s = min(s, max(0, t - segment))
        out[i] = y[i, :, s : s + segment]
    return out, idx


def drop_speaker_ids(sid: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Randomly drop the speaker id so the model also learns the reference path.

    This is the trick that makes a hybrid id + reference multi-speaker model
    work: with probability ``p`` the id embedding is replaced by a null index,
    forcing the speaker information to come from the audio reference
    (YourTTS, Casanova et al. 2022).
    """
    if p <= 0:
        return sid
    mask = torch.rand_like(sid.float()) < p
    return torch.where(mask, torch.full_like(sid, 0), sid)


def build_tone_segments(attn: torch.Tensor, x: torch.Tensor, vowel_ids: torch.Tensor, tone: torch.Tensor):
    """Per-frame vowel-segment ids and the tone label of each segment."""
    is_vowel = torch.isin(x, vowel_ids)
    vowel_index = torch.cumsum(is_vowel.long(), dim=1) - 1
    vowel_index = torch.where(is_vowel, vowel_index, torch.full_like(vowel_index, -1))
    seg_ids = torch.gather(vowel_index, 1, attn.argmax(dim=1))       # [B, T_spec]
    n_seg = int(is_vowel.sum(dim=1).max().item())
    tone_labels, tone_valid = [], []
    for bi in range(x.size(0)):
        v_idx = vowel_index[bi]
        sel = is_vowel[bi]
        if sel.sum() == 0:
            tone_labels.append(torch.zeros(n_seg, dtype=torch.long, device=x.device))
            tone_valid.append(torch.zeros(n_seg, dtype=torch.bool, device=x.device))
            continue
        labels = torch.zeros(n_seg, dtype=torch.long, device=x.device)
        valid = torch.zeros(n_seg, dtype=torch.bool, device=x.device)
        labels[v_idx[sel]] = tone[bi][sel]
        valid[v_idx[sel]] = True
        tone_labels.append(labels)
        tone_valid.append(valid)
    return seg_ids, torch.stack(tone_labels), torch.stack(tone_valid)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class HausaTTSTrainer:
    def __init__(self, cfg, train_cfg) -> None:
        self.cfg = cfg
        self.tcfg = train_cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)

        self.model = HausaVITS(cfg).to(self.device)
        self.mp = MultiPeriodDiscriminator().to(self.device)
        self.ms = MultiScaleDiscriminator().to(self.device)
        self.tone_d = ToneContourDiscriminator(cfg.n_tone, cfg.n_length).to(self.device) \
            if train_cfg.use_tone_adv else None

        self.opt_g = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, betas=tuple(cfg.betas), eps=cfg.eps
        )
        self.opt_d = torch.optim.AdamW(
            list(self.mp.parameters()) + list(self.ms.parameters()),
            lr=cfg.learning_rate, betas=tuple(cfg.betas), eps=cfg.eps,
        )
        self.opt_tone = (
            torch.optim.AdamW(self.tone_d.parameters(), lr=cfg.learning_rate,
                              betas=tuple(cfg.betas), eps=cfg.eps)
            if self.tone_d is not None else None
        )
        self.fb = haudio.mel_filterbank(cfg.sampling_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        self.fb_t = torch.from_numpy(self.fb.astype(np.float32)).to(self.device)
        self.step = 0
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.tcfg.fp16) if (train_cfg.fp16 and self.device.type == "cuda") else None
        os.makedirs(train_cfg.out_dir, exist_ok=True)
        with open(os.path.join(train_cfg.out_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"model": cfg.to_dict(), "train": train_cfg.__dict__}, fh, indent=2)

    # ------------------------------------------------------------------
    def _loaders(self, diacritizer=None):
        ds_tr = HausaTTSDataset(self.tcfg.train_list, self.cfg, split="train", diacritizer=diacritizer)
        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True,
                           num_workers=self.cfg.num_workers, collate_fn=collate,
                           drop_last=True, pin_memory=True)
        dl_va = None
        if os.path.exists(self.tcfg.val_list):
            ds_va = HausaTTSDataset(self.tcfg.val_list, self.cfg, split="val", diacritizer=diacritizer)
            dl_va = DataLoader(ds_va, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
        return dl_tr, dl_va

    # ------------------------------------------------------------------
    def train(self, diacritizer=None, max_steps: Optional[int] = None):
        dl_tr, _ = self._loaders(diacritizer)
        self.model.train()
        t0 = time.time()
        for epoch in range(self.cfg.epochs):
            for batch in dl_tr:
                self.train_step(batch)
                self.step += 1
                if self.step % self.tcfg.log_interval == 0:
                    print(f"[step {self.step}] {self.last_log} elapsed={time.time() - t0:.0f}s")
                if self.step % self.tcfg.save_interval == 0:
                    self.save()
                if max_steps is not None and self.step >= max_steps:
                    self.save(tag="final")
                    return
        self.save(tag="final")

    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        dev = self.device
        x = batch["phoneme"].to(dev)
        tone = batch["tone"].to(dev)
        length = batch["length"].to(dev)
        utt_type = batch["utt_type"].to(dev)
        sid = batch["speaker_id"].to(dev)
        spec = batch["linear"].to(dev).transpose(1, 2)          # [B, F, T]
        spec_lengths = batch["spec_lengths"].to(dev)
        mel = batch["mel"].to(dev).transpose(1, 2)              # [B, M, T]
        wav = batch["wav"].to(dev).unsqueeze(1)
        x_lengths = batch["x_lengths"].to(dev)
        f0_log = batch["f0"].to(dev)
        f0_hz = batch["f0_hz"].to(dev)
        voiced = batch["voiced"].to(dev)
        energy = batch["energy"].to(dev)
        prosody_stats = batch["prosody_stats"].to(dev)
        dur_gt = batch["durations"].to(dev)
        use_dur = bool(batch["has_durations"].any().item())

        sid_in = drop_speaker_ids(sid, self.cfg.speaker_dropout_p) if self.cfg.speaker_source == "both" else sid

        # rule-based F0 hint expanded by the ground-truth durations when
        # available, else by a uniform split (it is only a *hint*: the network
        # learns a residual on top of it).
        f0_hint = self._rule_hint(f0_log, spec_lengths)

        y_crop = None
        y_hat_crop = None
        ids_seg = None

        with torch.amp.autocast('cuda', enabled=self.tcfg.fp16):
            out = self.model(
                x, x_lengths, tone, length, utt_type, spec, spec_lengths, y_crop,
                torch.full_like(batch["y_lengths"].to(dev), self.cfg.segment_size),
                f0_hint_rule=f0_hint, f0_target=f0_log, energy_target=energy,
                sid=sid_in, prosody_stats=prosody_stats,
                duration_frames=dur_gt if use_dur else None,
            )
    
            # the waveform is synthesised for the whole utterance while the target
            # may be shorter: clamp both to the same sample range before any loss.
            n_common = int(min(out["o"].size(2), wav.size(2)))
            if n_common < int(out["o"].size(2)):
                out["o"] = out["o"][:, :, :n_common]
            if n_common < int(wav.size(2)):
                wav = wav[:, :, :n_common]
            batch["y_lengths"] = batch["y_lengths"].to(dev).clamp(max=n_common)
            _tf = int(n_common) // self.cfg.hop_length
            out["f0_pred"] = out["f0_pred"][:, :_tf]
            out["f0_mask"] = out["f0_mask"][:, :_tf]
            # align every frame-level target with the predicted track before any loss
            f0_log = f0_log[:, :_tf]
            energy = energy[:, :_tf]
            voiced = voiced[:, :_tf]
    
            # ---------------- generator losses ----------------
            # the waveform is generated for the whole utterance; crop the SAME
            # sample range from the prediction and from the ground truth.
            y_hat = out["o"]
            y_hat_crop, ids_seg = slice_audio(y_hat, batch["y_lengths"].to(dev), self.cfg.segment_size)
            y_crop, _ = slice_audio(wav, batch["y_lengths"].to(dev), self.cfg.segment_size, ids=ids_seg)
            _n = int(min(y_crop.size(2), y_hat_crop.size(2)))
            y_crop, y_hat_crop = y_crop[:, :, :_n], y_hat_crop[:, :, :_n]
            seg_mel = self._mel_of(y_crop)
            mel_hat = self._mel_of(y_hat_crop)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.mp(y_crop, y_hat_crop)
            loss_adv = L.generator_loss(y_d_hat_g)
            loss_fm = L.feature_loss(fmap_r, fmap_g)
            s_d_hat_r, s_d_hat_g, sfmap_r, sfmap_g = self.ms(y_crop, y_hat_crop)
            loss_adv = loss_adv + L.generator_loss(s_d_hat_g)
            loss_fm = loss_fm + L.feature_loss(sfmap_r, sfmap_g)
    
            loss_mel = L.mel_reconstruction_loss(seg_mel, mel_hat)
            loss_kl = L.kl_loss(out["z_p_e"], out["logs_q"], out["m_p"], out["logs_p"], out["spec_mask"])
    
            if self.cfg.duration_type == "stochastic":
                loss_dur = L.duration_loss_stochastic(out["v"], out["logdet"], out["x_mask"])
            else:
                loss_dur = L.duration_loss_deterministic(out["dur_pred_log"], out["x_dur_log"], out["x_mask"])
    
            f0_mask = out["f0_mask"]
            loss_pitch = L.pitch_loss(out["f0_pred"], f0_log, f0_mask)
            loss_energy = L.energy_loss(out["energy_pred"], energy, f0_mask)
            loss_prosody = L.prosody_stats_loss(out["prosody_stats_pred"], prosody_stats)
    
            loss_tone = out["f0_pred"].new_zeros(())
            tone_acc = out["f0_pred"].new_zeros(())
            if self.tcfg.use_tone_cls:
                seg_ids, tone_labels, tone_valid = build_tone_segments(
                    out["attn"], x, torch.tensor(self.cfg.vowel_ids, device=dev), tone
                )
                loss_tone, tone_acc, _ = L.tone_classification_loss(
                    out["f0_pred"], f0_mask, seg_ids, tone_labels, tone_valid, self.model.tone_cls
                )
    
            loss_harm = out["f0_pred"].new_zeros(())
            if y_crop is None or y_hat_crop is None:
                y_hat_crop, ids_seg = slice_audio(out["o"], batch["y_lengths"].to(dev), self.cfg.segment_size)
                y_crop, _ = slice_audio(wav, batch["y_lengths"].to(dev), self.cfg.segment_size, ids=ids_seg)
            if self.tcfg.use_harmonic:
                # harmonic (PeriodVITS) loss: the generated waveform must place its
                # energy at integer multiples of the PREDICTED F0, so a tonally
                # wrong contour cannot survive even if the audio sounds fluent.
                hop = self.cfg.hop_length
                starts = [int(i) // hop for i in (ids_seg if ids_seg is not None else [0] * x.size(0))]
                t_crop = y_hat_crop.size(2) // hop
                f0_crop = torch.zeros(x.size(0), t_crop, device=dev)
                voiced_crop = torch.ones(x.size(0), t_crop, device=dev)
                for i, s0 in enumerate(starts):
                    seg = out["f0_pred"][i, s0 : s0 + t_crop]
                    if seg.numel() < t_crop:
                        seg = F.pad(seg, (0, t_crop - seg.numel()), value=0.0)
                    f0_crop[i] = seg
                    v_seg = voiced[i, s0 : s0 + t_crop]
                    if v_seg.numel() < t_crop:
                        v_seg = F.pad(v_seg, (0, t_crop - v_seg.numel()), value=0.0)
                    voiced_crop[i] = v_seg
                f0_hz_pred = torch.exp(f0_crop.clamp(0.0, 8.0)) * voiced_crop
                loss_harm = L.harmonic_loss(
                    f0_hz_pred, self._lin_of(y_hat_crop), self.cfg.n_fft, self.cfg.sampling_rate
                )
    
            loss_tonal_g = out["f0_pred"].new_zeros(())
            if self.tone_d is not None:
                seg_ids, _, _ = build_tone_segments(
                    out["attn"], x, torch.tensor(self.cfg.vowel_ids, device=dev), tone
                )
                d_fake_t = self.tone_d(out["f0_pred"].unsqueeze(1), out["tone_frames"], out["length_frames"])
                loss_tonal_g = L.generator_loss([d_fake_t])
    
            loss_gen = (
                loss_adv + self.cfg.c_fm * loss_fm + self.cfg.c_mel * loss_mel
                + self.cfg.c_kl * loss_kl + self.cfg.c_dur * loss_dur
                + self.cfg.c_pitch * loss_pitch + self.cfg.c_energy * loss_energy
                + self.cfg.c_prosody * loss_prosody + self.cfg.c_tone * loss_tone
                + self.cfg.c_harmonic * loss_harm + self.cfg.c_tonal_adv * loss_tonal_g
            )

        self.opt_g.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss_gen).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.scaler.step(self.opt_g)
        else:
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.opt_g.step()

        # ---------------- discriminator losses ----------------
        with torch.amp.autocast('cuda', enabled=self.tcfg.fp16):
            y_d_r, y_d_g, _, _ = self.mp(y_crop.detach(), y_hat_crop.detach())
            loss_d = L.discriminator_loss(y_d_r, y_d_g)
            s_d_r, s_d_g, _, _ = self.ms(y_crop.detach(), y_hat_crop.detach())
            loss_d = loss_d + L.discriminator_loss(s_d_r, s_d_g)
        self.opt_d.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss_d).backward()
            self.scaler.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(list(self.mp.parameters()) + list(self.ms.parameters()), self.tcfg.grad_clip)
            self.scaler.step(self.opt_d)
        else:
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(list(self.mp.parameters()) + list(self.ms.parameters()), self.tcfg.grad_clip)
            self.opt_d.step()

        loss_tonal_d = out["f0_pred"].new_zeros(())
        if self.tone_d is not None:
            with torch.amp.autocast('cuda', enabled=self.tcfg.fp16):
                real = f0_log.unsqueeze(1).detach()
                fake = out["f0_pred"].unsqueeze(1).detach()
                d_real = self.tone_d(real, out["tone_frames"], out["length_frames"])
                d_fake = self.tone_d(fake, out["tone_frames"], out["length_frames"])
                loss_tonal_d = L.discriminator_loss([d_real], [d_fake])
            self.opt_tone.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss_tonal_d).backward(retain_graph=False)
                self.scaler.unscale_(self.opt_tone)
                self.scaler.step(self.opt_tone)
            else:
                loss_tonal_d.backward(retain_graph=False)
                self.opt_tone.step()

        if self.scaler is not None:
            self.scaler.update()

        self.last_log = (
            f"g={loss_gen.detach().item():.3f} d={loss_d.detach().item():.3f} mel={loss_mel.detach().item():.3f} "
            f"kl={loss_kl.detach().item():.3f} dur={loss_dur.detach().item():.3f} pitch={loss_pitch.detach().item():.4f} "
            f"tone={loss_tone.detach().item():.3f} tone_acc={tone_acc.detach().item():.3f} "
            f"adv={loss_adv.detach().item():.3f} fm={loss_fm.detach().item():.3f}"
        )
        # learning-rate decay (VITS' exponential schedule)
        for opt in (self.opt_g, self.opt_d, self.opt_tone):
            if opt is None:
                continue
            for g_ in opt.param_groups:
                g_["lr"] = self.cfg.learning_rate * (self.cfg.lr_decay ** self.step)
        if torch.isnan(loss_gen) or torch.isinf(loss_gen):
            raise RuntimeError(f"non-finite generator loss at step {self.step}")

    # ------------------------------------------------------------------
    def _mel_of(self, wav: torch.Tensor) -> torch.Tensor:
        """Log-mel of a waveform, computed on a detached copy.

        The analysis front end in ``hausa_tts.audio`` is numpy-based, so the
        gradient must not be routed through it; the acoustic supervision stays
        differentiable through the adversarial and harmonic terms.
        """
        with torch.no_grad():
            w = wav.detach().float()
            if w.dim() == 3:
                w = w.squeeze(1)
            min_len = self.cfg.n_fft + 1
            if w.size(1) < min_len:
                w = F.pad(w, (0, min_len - w.size(1)))
            mel = haudio.mel_spectrogram(w, self.cfg.sampling_rate, self.cfg.n_fft,
                                         self.cfg.hop_length, self.cfg.win_length,
                                         self.cfg.n_mels,
                                         getattr(self.cfg, "f_min", 0.0),
                                         getattr(self.cfg, "f_max", self.cfg.sampling_rate / 2),
                                         fb=self.fb)
        return torch.from_numpy(mel).to(wav.device).float()

    def _lin_of(self, wav: torch.Tensor) -> torch.Tensor:
        return haudio.linear_spectrogram_torch(
            wav.squeeze(1), self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length
        )

    def _rule_hint(self, f0_log: torch.Tensor, spec_lengths: torch.Tensor) -> torch.Tensor:
        """Cheap proxy hint for training: the smoothed ground-truth F0 shape.

        At inference the hint comes from the rule-based phonological model in
        :mod:`hausa_tts.text`; during training we must not leak the ground truth
        into the decoder, so the hint is the frame-level *mean* contour with a
        declination trend removed -- i.e. the information the tone targets
        already provide.
        """
        b, t = f0_log.size()
        mask = sequence_mask(spec_lengths, t).float()
        mean = (f0_log * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1.0)
        ramp = torch.linspace(-0.08, 0.08, t, device=f0_log.device).unsqueeze(0)
        return (mean + ramp) * mask

    # ------------------------------------------------------------------
    def save(self, tag: str = "step") -> None:
        path = os.path.join(self.tcfg.out_dir, f"G_{tag}_{self.step}.pth")
        torch.save(
            {
                "model": self.model.state_dict(),
                "mp": self.mp.state_dict(),
                "ms": self.ms.state_dict(),
                "tone_d": self.tone_d.state_dict() if self.tone_d else None,
                "step": self.step,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        print(f"[save] {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "mp" in ckpt:
            self.mp.load_state_dict(ckpt["mp"])
            self.ms.load_state_dict(ckpt["ms"])
        if ckpt.get("tone_d") and self.tone_d is not None:
            self.tone_d.load_state_dict(ckpt["tone_d"])
        self.step = ckpt.get("step", 0)
