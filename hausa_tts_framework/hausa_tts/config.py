"""Configuration objects for the Hausa TTS framework."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence

from . import text as ha
from .modules import ProsodyEncoder


@dataclass
class HausaVITSConfig:
    """Every design decision of the framework in one auditable place."""

    # ---------------- data ----------------
    data_dir: str = ""              # prefix used to resolve relative wav paths
    sampling_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = 8000.0
    spec_channels: int = 513          # n_fft // 2 + 1
    segment_size: int = 8192          # training crop (samples)
    f0_min_hz: float = 60.0
    f0_max_hz: float = 500.0

    # ---------------- text ----------------
    n_vocab: int = field(default_factory=lambda: len(ha.PHONEMES))
    n_tone: int = field(default_factory=lambda: len(ha.TONES))
    n_length: int = field(default_factory=lambda: len(ha.LENGTHS))
    n_utt_type: int = field(default_factory=lambda: len(ha.UTTERANCE_TYPES))
    vowel_ids: Sequence[int] = field(
        default_factory=lambda: [ha.PHONEME2ID[v] for v in ha.VOWEL_GRAPHEMES]
    )
    silence_ids: Sequence[int] = field(
        default_factory=lambda: [ha.PHONEME2ID[ha.SIL_TALK], ha.PHONEME2ID[ha.SIL_PAUSE]]
    )

    # ---------------- architecture ----------------
    hidden_channels: int = 192
    filter_channels: int = 768
    attn_hidden_scale: int = 1
    inter_channels: int = 192
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    window_size: int = 4
    n_flows: int = 4

    duration_type: str = "stochastic"     # "deterministic" | "stochastic"
    n_flows_dur: int = 4
    length_scaled_duration: bool = True   # [HAUSA] per-(tone, length) duration bias
    align_prior: bool = True              # [HAUSA] vowel/pause alignment prior

    prosody_channels: int = 64
    prosody_emb: int = 32                 # size of the prosody vector added to the decoder
    use_prosody_latent: bool = True
    pitch_hidden: int = 192
    pitch_layers: int = 3

    use_f0_hint: bool = True              # [HAUSA] explicit pitch injection in the vocoder
    f0_hint_dim: int = 64

    upsample_initial_channel: int = 512
    upsample_rates: Sequence[int] = (8, 8, 2, 2)
    upsample_kernel_sizes: Sequence[int] = (16, 16, 4, 4)
    resblock: str = "1"
    resblock_kernel_sizes: Sequence[int] = (3, 7, 11)
    resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

    # ---------------- speakers ----------------
    # "none"  -> single speaker
    # "id"    -> multi speaker, learned embedding per speaker
    # "reference" -> zero-shot multi speaker from reference audio
    # "both"  -> hybrid (recommended: 0.5 probability of dropping the id during
    #            training so the model also learns the reference path)
    speaker_source: str = "id"
    n_speakers: int = 1
    gin_channels: int = 256
    speaker_enc_channels: int = 128
    speaker_dropout_p: float = 0.5

    # ---------------- optimisation ----------------
    learning_rate: float = 2e-4
    betas: Sequence[float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875
    batch_size: int = 16
    epochs: int = 500
    warmup_epochs: int = 5

    # ---------------- loss weights ----------------
    c_mel: float = 45.0
    c_kl: float = 1.0
    c_dur: float = 1.0
    c_pitch: float = 2.0       # [HAUSA] continuous F0 loss
    c_energy: float = 1.0
    c_prosody: float = 0.5
    c_tone: float = 1.0        # [HAUSA] lexical-tone classification loss
    c_tonal_adv: float = 0.5   # [HAUSA] tonal adversarial critic
    c_harmonic: float = 0.5    # [HAUSA] harmonic (PeriodVITS) loss
    c_fm: float = 2.0

    # ---------------- prosody parameters ----------------
    h_level: float = 0.00
    l_level: float = -0.34
    r_level: float = -0.16
    f_level: float = -0.10
    declination: float = 0.16
    question_rise: float = 0.30
    wh_boost: float = 0.14

    # ---------------- misc ----------------
    seed: int = 1234
    device: str = "cuda"
    num_workers: int = 4

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HausaVITSConfig":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})

    # ------------------------------------------------------------------
    # convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def single_speaker(cls, **kw) -> "HausaVITSConfig":
        base = dict(speaker_source="none", n_speakers=1, gin_channels=0)
        base.update(kw)
        return cls(**base)

    @classmethod
    def multi_speaker(cls, n_speakers: int, **kw) -> "HausaVITSConfig":
        base = dict(speaker_source="id", n_speakers=n_speakers)
        base.update(kw)
        return cls(**base)

    @classmethod
    def zero_shot_multi_speaker(cls, n_speakers: int = 1, **kw) -> "HausaVITSConfig":
        base = dict(speaker_source="both", n_speakers=max(1, n_speakers))
        base.update(kw)
        return cls(**base)


@dataclass
class TrainConfig:
    """Training-loop settings kept separate from the model architecture."""

    data_dir: str = "data"
    train_list: str = "data/train.txt"
    val_list: str = "data/val.txt"
    out_dir: str = "runs/hausa_vits"
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    spec_cache: bool = True
    f0_cache: bool = True
    grad_clip: float = 1.0
    fp16: bool = False
    use_tone_adv: bool = True
    use_tone_cls: bool = True
    use_harmonic: bool = True
    tensorboard: bool = True
    seed: int = 1234
