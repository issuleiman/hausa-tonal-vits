"""Hausa VITS: a tonal, length-aware extension of VITS for single- and
multi-speaker Hausa text-to-speech.

Quick start
-----------
>>> from hausa_tts import HausaVITSConfig, HausaVITS, HausaTTS
>>> cfg = HausaVITSConfig.single_speaker()
>>> model = HausaVITS(cfg)

See ``README.md`` for the full pipeline, the linguistics it implements, and how
to train on BibleTTS / NaijaVoices / Common Voice data.
"""

from .config import HausaVITSConfig, TrainConfig
from .text import PHONEMES, PHONEME2ID, TONES, TONE2ID, LENGTHS, g2p, normalise
from .model import HausaVITS
from .inference import HausaTTS, SynthOptions, save_wav
from .data import HausaTTSDataset, collate
from .diacritizer import (
    LexiconDiacritizer,
    NGramDiacritizer,
    NeuralDiacritizer,
    build_diacritizer,
    train_diacritizer,
)

__version__ = "1.0.0"

__all__ = [
    "HausaVITSConfig",
    "TrainConfig",
    "HausaVITS",
    "HausaTTS",
    "SynthOptions",
    "save_wav",
    "HausaTTSDataset",
    "collate",
    "g2p",
    "normalise",
    "PHONEMES",
    "PHONEME2ID",
    "TONES",
    "TONE2ID",
    "LENGTHS",
    "LexiconDiacritizer",
    "NGramDiacritizer",
    "NeuralDiacritizer",
    "build_diacritizer",
    "train_diacritizer",
]
