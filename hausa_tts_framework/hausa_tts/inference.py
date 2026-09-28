"""Inference API: single-speaker and multi-speaker synthesis.

Three use-modes:

**1. Single speaker** -- one voice, tone-correct by construction::

    tts = HausaTTS.from_pretrained("runs/hausa_vits/G_final.pth")
    wav = tts.speak("Yarinƙa tana̋ son abinci.")

**2. Multi speaker (seen speakers)** -- pass ``speaker="spk_03"``.

**3. Multi speaker (unseen speakers, zero-shot)** -- pass a few seconds of
reference audio; it is embedded with the ECAPA-TDNN speaker encoder and
conditions the model (YourTTS, Casanova et al. 2022).

Prosody control: ``rate``, ``pitch`` (in log-F0 units), ``energy``,
``emotion="question"`` (yes/no final rise), plus explicit
``utt_type="statement"|"yesno"|"wh"``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import audio as haudio
from . import text as ha
from .config import HausaVITSConfig
from .data import load_wav, normalise_audio
from .model import HausaVITS
from .modules import sequence_mask


@dataclass
class SynthOptions:
    speaker: Optional[str] = None
    reference_wav: Optional[str] = None
    rate: float = 1.0
    pitch: float = 0.0           # log-domain offset applied to the F0 hint
    pitch_range: float = 1.0     # >1 exaggerates tone excursions
    energy: float = 1.0
    utt_type: Optional[str] = None
    noise_scale: float = 0.667
    noise_scale_dur: float = 0.8
    seed: Optional[int] = None
    durations: Optional[Sequence[int]] = None
    vibrato: float = 0.0


class HausaTTS:
    """High-level wrapper around :class:`HausaVITS`."""

    def __init__(self, model: HausaVITS, cfg: HausaVITSConfig, speakers: Optional[Dict[str, int]] = None):
        self.model = model.eval()
        self.cfg = cfg
        self.speakers = speakers or {}

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, ckpt_path: str, device: str = "cpu",
                        speakers: Optional[Dict[str, int]] = None) -> "HausaTTS":
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = HausaVITSConfig.from_dict(ckpt.get("config", {}))
        model = HausaVITS(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        spk = speakers
        if spk is None and "speakers" in ckpt:
            spk = ckpt["speakers"]
        return cls(model, cfg, spk)

    # ------------------------------------------------------------------
    def preprocess(self, text: str, utt_type: Optional[str] = None) -> Tuple[torch.Tensor, ...]:
        tokens = ha.g2p(text)
        enc = ha.encode(tokens)
        utt = utt_type or _guess_utt(text)
        x = torch.tensor([enc["phoneme"]], dtype=torch.long)
        tone = torch.tensor([enc["tone"]], dtype=torch.long)
        length = torch.tensor([enc["length"]], dtype=torch.long)
        utt_id = torch.tensor([ha.UTT2ID.get(utt, 0)], dtype=torch.long)
        return x, tone, length, utt_id, tokens

    # ------------------------------------------------------------------
    @torch.no_grad()
    def speak(self, text: str, opts: SynthOptions = None, **kw) -> np.ndarray:
        if opts is None:
            opts = SynthOptions(**kw)
        if opts.seed is not None:
            torch.manual_seed(opts.seed)
        x, tone, length, utt_id, tokens = self.preprocess(text, opts.utt_type)
        dev = next(self.model.parameters()).device
        x = x.to(dev)
        tone = tone.to(dev)
        length = length.to(dev)
        utt_id = utt_id.to(dev)

        sid = None
        if opts.speaker is not None:
            if opts.speaker not in self.speakers:
                raise KeyError(f"unknown speaker {opts.speaker!r}; known: {sorted(self.speakers)}")
            sid = torch.tensor([self.speakers[opts.speaker]], dtype=torch.long, device=dev)

        ref_mel = ref_mask = None
        if opts.reference_wav:
            ref = load_wav(opts.reference_wav, self.cfg.sampling_rate)
            ref = normalise_audio(ref)
            fb = haudio.mel_filterbank(self.cfg.sampling_rate, self.cfg.n_fft, self.cfg.n_mels,
                                       self.cfg.fmin, self.cfg.fmax)
            mel = haudio.mel_spectrogram(ref, self.cfg.sampling_rate, self.cfg.n_fft,
                                         self.cfg.hop_length, self.cfg.win_length, self.cfg.n_mels,
                                         self.cfg.fmin, self.cfg.fmax, fb=fb)
            ref_mel = torch.from_numpy(mel.T).float().unsqueeze(0).transpose(1, 2).to(dev)
            ref_mask = torch.ones(1, 1, ref_mel.size(2), device=dev)

        dur_override = None
        if opts.durations is not None:
            d = torch.tensor([list(opts.durations)], dtype=torch.long, device=dev)
            if d.size(1) < x.size(1):
                d = torch.nn.functional.pad(d, (0, x.size(1) - d.size(1)))
            dur_override = d

        audio, attn, f0, durations = self.model.infer(
            x, tone, length, utt_id, sid=sid, reference_mel=ref_mel,
            reference_mel_mask=ref_mask, duration_override=dur_override,
            length_scale=1.0 / max(opts.rate, 1e-3),
            noise_scale=opts.noise_scale, noise_scale_dur=opts.noise_scale_dur,
            f0_scale=opts.pitch_range, energy_scale=opts.energy,
        )
        wav = audio.squeeze().cpu().numpy()
        if opts.pitch != 0.0:
            wav = _shift_pitch_simple(wav, self.cfg.sampling_rate, opts.pitch)
        return wav

    # ------------------------------------------------------------------
    @torch.no_grad()
    def speak_with_alignment(self, text: str, **kw):
        """Return ``(wav, attn, f0_hz, durations, tokens)`` for inspection."""
        x, tone, length, utt_id, tokens = self.preprocess(text, kw.get("utt_type"))
        audio, attn, f0, durations = self.model.infer(
            x, tone, length, utt_id,
            length_scale=1.0 / max(kw.get("rate", 1.0), 1e-3),
            noise_scale=kw.get("noise_scale", 0.667),
            noise_scale_dur=kw.get("noise_scale_dur", 0.8),
        )
        return audio.squeeze().cpu().numpy(), attn.cpu().numpy(), f0.cpu().numpy(), \
            durations.cpu().numpy(), tokens


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def _guess_utt(text: str) -> str:
    t = text.strip()
    if not t.endswith("?"):
        return "statement"
    first = t.lower().split()[0] if t.split() else ""
    wh = ("wa", "ina", "yaushe", "me", "wanne", "nawa", "ɗaya", "yaya")
    return "wh" if any(first.startswith(w) for w in wh) else "yesno"


def _shift_pitch_simple(wav: np.ndarray, sr: int, semitones_log: float) -> np.ndarray:
    """Placeholder pitch shift in the log-F0 domain.

    A production system would use a proper vocoder-based shift; here we
    resample-then-stretch so the API is complete without extra dependencies.
    """
    factor = float(np.exp(semitones_log))
    if abs(factor - 1.0) < 1e-3:
        return wav
    n_out = int(len(wav) / factor)
    x_in = np.linspace(0.0, 1.0, len(wav), endpoint=False)
    x_out = np.linspace(0.0, 1.0, max(1, n_out), endpoint=False)
    return np.interp(x_out, x_in, wav).astype(np.float32)


def save_wav(path: str, wav: np.ndarray, sr: int) -> str:
    """Write a wav file (``soundfile`` if present, else the stdlib)."""
    wav = np.asarray(wav, dtype=np.float32)
    try:
        import soundfile as sf

        sf.write(path, wav, sr)
        return path
    except ImportError:
        import wave

        pcm = np.clip(wav, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return path
