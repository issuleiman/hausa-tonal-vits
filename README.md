# Hausa Tonal VITS

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tone-aware, single-stage end-to-end Text-to-Speech (TTS) framework for the **Hausa language**, extending the [VITS](https://arxiv.org/abs/2106.06103) (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture. 

It introduces explicit lexical-tone and vowel-length conditioning, a rule-based Hausa prosody engine, a tone-restoration front-end (diacritizer), and a pitch-hint-conditioned HiFi-GAN decoder.

---

## 🌟 Why This Extension Is Needed

Hausa (Chadic, spoken by over 80 million people across West and Central Africa) is a **tonal language** distinguishing **High (á)**, **Low (à)**, and **Falling (â)** tones, along with contrastive **vowel length** (short vs. long).

Standard Boko orthography does not mark tone or vowel length in everyday writing. Because lexical tone is the primary cue to word identity and meaning in Hausa, standard TTS systems (like vanilla VITS or Tacotron) suffer from:
1. **Semantic Ambiguity**: Words with identical spellings but different meanings (tonal minimal pairs) cannot be distinguished.
2. **Unnatural Pitch Contours**: Missing sentence declination, question intonation, and downdrift.
3. **Vowel Duration Neutralization**: Collapsing the essential long/short vowel length contrast.

---

## 🚀 The 7 Tonal & Prosodic Innovations

1. **Three-Stream FiLM Text Encoder**: Separates phonemes, tones, and vowel lengths into distinct parallel input streams rather than expanding the vocabulary combinatorially.
2. **Residual Pitch on Phonological F0 Targets**: Neural pitch predictor learns *corrections* on top of explicit Hausa phonological rules (declination, downdrift, yes/no question rise, wh-question boost).
3. **Per-(Tone, Length) Duration Bias**: Learnable scale parameters in the duration predictor guarantee that the model cannot collapse the short/long vowel distinction.
4. **Tone-Conditioned Stochastic Durations**: Normalizing flow duration sampling conditioned on tone/length features.
5. **F0-Conditioned HiFi-GAN Decoder**: Sinusoidal F0 feature injection at every upsampling stage of the neural vocoder.
6. **Tone Contour Discriminator**: An adversarial critic judging generated pitch contours against intended tone sequences.
7. **Differentiable Tone Classification Loss**: Vowel-segment F0 pooling supervised by cross-entropy against true tone labels.

---

## 📁 Repository Structure

```
hausa-tonal-vits/
├── configs/
│   ├── single_speaker.yaml       # Single-speaker model & training configuration
│   └── multi_speaker.yaml        # Multi-speaker configuration (id & zero-shot)
├── hausa_tts_framework/
│   ├── hausa_tts/
│   │   ├── __init__.py           # Package exports & public API
│   │   ├── text.py               # G2P, phoneme inventory, post-lexical rules, F0 targets
│   │   ├── diacritizer.py        # Lexicon, n-gram, and neural BiLSTM tone restoration
│   │   ├── audio.py              # STFT, mel filterbank, YIN F0, prosody stats
│   │   ├── data.py               # Dataset loader, smart batching & collator
│   │   ├── config.py             # HausaVITSConfig and TrainConfig dataclasses
│   │   ├── model.py              # HausaVITS orchestrator (forward & infer)
│   │   ├── modules.py            # Text encoder, WN, duration/pitch/energy heads
│   │   ├── flows.py              # Normalizing flows & spline duration predictor
│   │   ├── vocoder.py            # HiFi-GAN generator & multi-period discriminators
│   │   ├── losses.py             # VITS, harmonic, and tonal loss suite
│   │   ├── train.py              # Full adversarial trainer & checkpoint manager
│   │   ├── inference.py          # High-level synthesis API & WAV generation
│   │   └── metrics.py            # MCD, F0 RMSE/correlation, Tone Accuracy, RTF
│   └── tools/
│       ├── smoke_test.py         # 21-check end-to-end self-test on dummy data
│       └── prepare_dataset.py    # Dataset validation, audio checks, train/val split
├── pyproject.toml                # Build system & package specification
├── setup.py                      # Pip installation script
└── requirements.txt              # Core Python dependencies
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/issuleiman/hausa-tonal-vits.git
cd hausa-tonal-vits

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

Verify your installation by running the 21-check verification test:
```bash
python hausa_tts_framework/tools/smoke_test.py
```

---

## 📊 Dataset Preparation

Prepare your data as a pipe-separated metadata file (`wav_path|text`):

```
path/to/sample1.wav|Yaushe za ka dawo?
path/to/sample2.wav|Ina son koyon harshen Hausa.
```

Run the preparation tool to validate audio, normalize text, filter durations, and generate train/val splits:
```bash
python hausa_tts_framework/tools/prepare_dataset.py \
  --input metadata.csv \
  --output_dir data/ \
  --val_ratio 0.05 \
  --min_duration 0.5 \
  --max_duration 12.0
```

---

## 🏋️ Training

### Single-Speaker Training
```bash
python -m hausa_tts.train --config configs/single_speaker.yaml
```

### Multi-Speaker / Custom Options
```bash
python -m hausa_tts.train \
  --config configs/multi_speaker.yaml \
  --train_list data/train.txt \
  --val_list data/val.txt \
  --out_dir runs/my_experiment \
  --batch_size 16 \
  --fp16
```

---

## 🎙️ Inference & Synthesis

Synthesizing speech from text with automatic tone diacritization:

```python
from hausa_tts import HausaTTS, SynthOptions, save_wav

# Load trained checkpoint
tts = HausaTTS.from_pretrained("runs/hausa_vits/G_step_10000.pth")

# Synthesize speech
audio = tts.speak(
    "Yaushe za ka dawo gida?",
    opts=SynthOptions(
        rate=1.0,         # Speed factor
        pitch=0.0,        # Log-F0 offset
        pitch_range=1.0   # Pitch excursion scale
    )
)

# Save output waveform
save_wav(audio, "output.wav", sample_rate=22050)
```

---

## 📈 Evaluation Metrics

The framework includes comprehensive evaluation tools in `hausa_tts.metrics`:
- **Mel Cepstral Distortion (MCD)**: Spectral distance (dB) between natural and synthesized audio.
- **Log-F0 RMSE & Pearson Correlation**: Accuracy of pitch contour across voiced regions.
- **Tone Accuracy & Tone Faithfulness**: Hausa-specific evaluation assessing whether synthesized vowel segments align with ground-truth tone targets.
- **Real-Time Factor (RTF)**: Synthesis speed metric.

---

## 📚 References & Datasets

- **VITS**: Kim, J., Kong, J., & Son, J. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*. ICML.
- **Hausa Phonology**: Newman, P. (2000). *The Hausa Language: An Encyclopedic Reference Grammar*. Yale University Press.
- **Open Speech Data**:
  - [BibleTTS Hausa](https://www.bibletts.org/)
  - [Mozilla Common Voice (Hausa)](https://commonvoice.mozilla.org/ha)
  - NaijaVoices Speech Corpus

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
