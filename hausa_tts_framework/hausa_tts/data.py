"""Dataset / batching for Hausa TTS.

Two supported input layouts.

**A. Simple (Hausa text + wav) -- ``filelist`` mode**

    ``train.txt`` with one entry per line::

        path/to/0001.wav|Yarinƙa tana̋ son abinci.|yesno

    The second field is Hausa text (tone marks optional -- if absent, the
    front-end falls back to the unmarked reading and, when a diacritizer is
    available, you can plug it in via ``diacritizer=``). The third field is an
    optional utterance type (``statement`` / ``yesno`` / ``wh``); if omitted it
    is guessed from the punctuation of the text.

**B. With durations -- ``durations`` mode**

    ``path/to/0001.wav|Yarinƙa tana son abinci.|statement|0 0 2 1 ...``

    where the fourth field holds one frame duration per phoneme, produced by an
    external aligner (e.g. the Montreal Forced Aligner, McAuliffe et al.). This
    is the recommended low-resource recipe: pre-aligned durations remove the
    alignment search from the critical path and dramatically speed up early
    training on small Hausa corpora.

Speaker handling: prefix the file name (not the text) with ``speaker_id``
(``spk1/0001.wav``) or provide a fifth field with the integer speaker id.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import audio as haudio
from . import text as ha


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read_filelist(path: str) -> List[Tuple[str, str, str, Optional[List[int]], Optional[int]]]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            wav = parts[0].strip()
            txt = parts[1].strip() if len(parts) > 1 else ""
            utt = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else _guess_utt(txt)
            durs = None
            if len(parts) > 3 and parts[3].strip():
                durs = [int(float(x)) for x in parts[3].split()]
            sid = None
            if len(parts) > 4 and parts[4].strip():
                sid = int(parts[4])
            items.append((wav, txt, utt, durs, sid))
    return items


def _guess_utt(txt: str) -> str:
    t = txt.strip()
    if t.endswith("?") and any(
        t.lower().split()[0].startswith(w) for w in ("wa", "ina", "yaushe", "me", "wanne", "nawa", "ɗaya")
    ):
        return "wh"
    if t.endswith("?"):
        return "yesno"
    return "statement"


def load_wav(path: str, sr: int) -> np.ndarray:
    """Read a wav file (``soundfile`` if available, else the stdlib ``wave``)."""
    try:
        import soundfile as sf

        wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = resample_linear(wav, file_sr, sr)
        return wav.astype(np.float32)
    except Exception:
        import wave

        with wave.open(path, "rb") as wf:
            n = wf.getnframes()
            data = wf.readframes(n)
            file_sr = wf.getframerate()
            width = wf.getsampwidth()
            ch = wf.getnchannels()
        if width == 2:
            arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 1:
            arr = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            arr = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width {width}")
        if ch > 1:
            arr = arr.reshape(-1, ch).mean(axis=1)
        if file_sr != sr:
            arr = resample_linear(arr, file_sr, sr)
        return arr.astype(np.float32)


def resample_linear(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    n_out = int(round(len(wav) * sr_out / sr_in))
    x_in = np.linspace(0.0, 1.0, len(wav), endpoint=False)
    x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_out, x_in, wav).astype(np.float32)


def normalise_audio(wav: np.ndarray, target_db: float = -23.0, eps: float = 1e-8) -> np.ndarray:
    """RMS normalisation + peak protection (read-speech corpora vary a lot)."""
    rms = float(np.sqrt(np.mean(wav**2) + eps))
    if rms < 1e-6:
        return wav
    gain = 10 ** (target_db / 20.0) / rms
    out = wav * gain
    peak = float(np.max(np.abs(out)) + eps)
    if peak > 1.0:
        out = out / peak * 0.99
    return out.astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    frame = int(0.02 * sr)
    if len(wav) < frame * 3:
        return wav
    n = len(wav) // frame
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-10)
    db = 20 * np.log10(rms / (rms.max() + 1e-10) + 1e-10)
    keep = np.where(db > -top_db)[0]
    if keep.size == 0:
        return wav
    start = max(0, (keep[0] - 2)) * frame
    end = min(n, keep[-1] + 3) * frame
    return wav[start:end]


# ---------------------------------------------------------------------------
# item container
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    wav_path: str
    text: str
    utt_type: str
    speaker_id: int
    durations: Optional[List[int]] = None


class HausaTTSDataset(Dataset):
    """Map-style dataset yielding the tensors the model consumes."""

    def __init__(
        self,
        filelist: str,
        cfg,
        *,
        split: str = "train",
        diacritizer: Optional[Callable[[str], str]] = None,
        phone_durations: Optional[Dict[str, List[int]]] = None,
        clean: bool = True,
        max_seconds: float = 20.0,
        min_seconds: float = 0.4,
    ) -> None:
        self.items = read_filelist(filelist)
        self.cfg = cfg
        self.split = split
        self.diacritizer = diacritizer
        self.phone_durations = phone_durations or {}
        self.clean = clean
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        self.fb = haudio.mel_filterbank(cfg.sampling_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        self._filter()

    def _filter(self) -> None:
        kept = []
        for wav, txt, utt, durs, sid in self.items:
            if not txt:
                continue
            kept.append((wav, txt, utt, durs, sid))
        self.items = kept

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        wav_path, raw_text, utt, durs, sid = self.items[index]
        if self.diacritizer is not None:
            raw_text = self.diacritizer(raw_text)
        tokens = ha.g2p(raw_text, syllable_marker=False)
        if self.phone_durations and wav_path in self.phone_durations:
            durs = self.phone_durations[wav_path]
        data_dir = getattr(self.cfg, "data_dir", "") or ""
        full_path = wav_path if os.path.isabs(wav_path) else os.path.join(data_dir, wav_path)
        wav = load_wav(full_path, self.cfg.sampling_rate)
        if self.clean:
            wav = trim_silence(wav, self.cfg.sampling_rate)
            wav = normalise_audio(wav)
        if not (self.min_seconds <= len(wav) / self.cfg.sampling_rate <= self.max_seconds):
            # keep the sample but clamp its length so batching stays sane
            max_len = int(self.max_seconds * self.cfg.sampling_rate)
            wav = wav[:max_len]

        mel = haudio.mel_spectrogram(
            wav, self.cfg.sampling_rate, self.cfg.n_fft, self.cfg.hop_length,
            self.cfg.win_length, self.cfg.n_mels, self.cfg.fmin, self.cfg.fmax, fb=self.fb,
        )
        spec = haudio.linear_spectrogram(wav, self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length)
        f0, voiced = haudio.yin_f0(wav, self.cfg.sampling_rate, self.cfg.hop_length,
                                   self.cfg.win_length, self.cfg.f0_min_hz, self.cfg.f0_max_hz)
        f0_log = haudio.f0_to_continuous(f0, voiced)
        energy = np.log(np.sqrt(np.mean(spec**2, axis=0)) + 1e-6)
        stats = haudio.f0_stats(f0, voiced, energy)

        # pad/extend frame-level targets to the spectrogram length
        n_frames = spec.shape[1]
        f0_log = _fit_length(f0_log, n_frames)
        energy = _fit_length(energy, n_frames)
        voiced = _fit_length(voiced.astype(np.float32), n_frames)
        # the YIN frame grid differs from the centre-padded STFT grid by one or
        # two frames: realign every frame-level track to the spectrogram length
        f0 = _fit_length(f0, n_frames)

        utt_id = ha.UTT2ID.get(utt, 0)
        enc = ha.encode(tokens)
        # phone durations (optional): one per token
        dur_tensor = torch.zeros(len(tokens), dtype=torch.long)
        if durs is not None and len(durs) == len(tokens):
            dur_tensor = torch.tensor(durs, dtype=torch.long)

        return {
            "phoneme": torch.tensor(enc["phoneme"], dtype=torch.long),
            "tone": torch.tensor(enc["tone"], dtype=torch.long),
            "length": torch.tensor(enc["length"], dtype=torch.long),
            "utt_type": torch.tensor(utt_id, dtype=torch.long),
            "speaker_id": torch.tensor(int(sid) if sid is not None else 0, dtype=torch.long),
            "durations": dur_tensor,
            "has_durations": torch.tensor(durs is not None, dtype=torch.bool),
            "mel": torch.from_numpy(mel.T.astype(np.float32)),          # [T, n_mels]
            "linear": torch.from_numpy(spec.T.astype(np.float32)),       # [T, F]
            "f0": torch.from_numpy(f0_log.astype(np.float32)),
            "f0_hz": torch.from_numpy(f0.astype(np.float32)),
            "voiced": torch.from_numpy(voiced.astype(np.float32)),
            "energy": torch.from_numpy(energy.astype(np.float32)),
            "prosody_stats": torch.from_numpy(stats.astype(np.float32)),
            "wav": torch.from_numpy(wav.astype(np.float32)),
            "text": raw_text,
            "wav_path": wav_path,
        }


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    m = len(arr)
    if m == n:
        return arr
    if m > n:
        return arr[:n]
    pad = n - m
    return np.pad(arr, (0, pad), mode="edge")


# ---------------------------------------------------------------------------
# collation
# ---------------------------------------------------------------------------


def collate(batch: List[Dict[str, torch.Tensor]], hop_length: int = 256) -> Dict[str, torch.Tensor]:
    """Pad a batch; every sequence keeps its own length tensor."""
    max_txt = max(b["phoneme"].size(0) for b in batch)
    max_spec = max(b["mel"].size(0) for b in batch)
    max_wav = max(b["wav"].size(0) for b in batch)
    n_mels = batch[0]["mel"].size(1)
    f_bins = batch[0]["linear"].size(1)
    b = len(batch)

    out: Dict[str, object] = {
        "phoneme": torch.zeros(b, max_txt, dtype=torch.long),
        "tone": torch.zeros(b, max_txt, dtype=torch.long),
        "length": torch.zeros(b, max_txt, dtype=torch.long),
        "durations": torch.zeros(b, max_txt, dtype=torch.long),
        "has_durations": torch.zeros(b, dtype=torch.bool),
        "utt_type": torch.zeros(b, dtype=torch.long),
        "speaker_id": torch.zeros(b, dtype=torch.long),
        "mel": torch.zeros(b, max_spec, n_mels),
        "linear": torch.zeros(b, max_spec, f_bins),
        "f0": torch.zeros(b, max_spec),
        "f0_hz": torch.zeros(b, max_spec),
        "voiced": torch.zeros(b, max_spec),
        "energy": torch.zeros(b, max_spec),
        "prosody_stats": torch.zeros(b, 8),
        "wav": torch.zeros(b, max_wav),
        "x_lengths": torch.zeros(b, dtype=torch.long),
        "spec_lengths": torch.zeros(b, dtype=torch.long),
        "y_lengths": torch.zeros(b, dtype=torch.long),
        "text": [item["text"] for item in batch],
        "wav_path": [item["wav_path"] for item in batch],
    }
    for i, item in enumerate(batch):
        t = item["phoneme"].size(0)
        # the YIN frame grid and the centre-padded STFT grid can differ by a
        # frame, so the frame-level slot follows the shortest track of the item
        s = min(item["mel"].size(0), item["f0"].size(0), item["f0_hz"].size(0),
                item["voiced"].size(0), item["energy"].size(0),
                item["linear"].size(0), item["mel"].size(0))
        w = item["wav"].size(0)
        out["phoneme"][i, :t] = item["phoneme"]
        out["tone"][i, :t] = item["tone"]
        out["length"][i, :t] = item["length"]
        out["durations"][i, :t] = item["durations"]
        out["has_durations"][i] = item["has_durations"]
        out["utt_type"][i] = item["utt_type"]
        out["speaker_id"][i] = item["speaker_id"]
        out["mel"][i, :s] = item["mel"]
        out["linear"][i, :s] = item["linear"]
        out["f0"][i, :s] = item["f0"]
        out["f0_hz"][i, :s] = item["f0_hz"]
        out["voiced"][i, :s] = item["voiced"]
        out["energy"][i, :s] = item["energy"]
        out["prosody_stats"][i] = item["prosody_stats"]
        out["wav"][i, :w] = item["wav"]
        out["x_lengths"][i] = t
        out["spec_lengths"][i] = s
        out["y_lengths"][i] = w
    out["y_lengths"] = torch.tensor([item["wav"].size(0) for item in batch], dtype=torch.long)
    return out  # type: ignore[return-value]
