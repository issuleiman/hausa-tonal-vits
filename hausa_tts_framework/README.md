# Hausa Tonal VITS

A single-stage end-to-end text-to-speech framework for **Hausa**, extending the
VITS architecture with explicit lexical-tone and vowel-length conditioning, a
rule-based Hausa prosody engine, a tone-restoration front end and a
pitch-hint-conditioned HiFi-GAN decoder. Supports single-speaker and
multi-speaker (id-embedding and zero-shot reference) training.

## Why an extension is needed
Hausa (Chadic, ~80M speakers) distinguishes High, Low and Falling tone, and
vowel length is phonemic. Standard Boko orthography marks neither, and lexical
tone is the primary cue to word identity. A plain VITS text encoder, which sees
a single grapheme stream, cannot recover tone that is not in the input.

## Modules
| file | role |
|---|---|
| `hausa_tts/phonemes.py` | phoneme inventory incl. ɓ ɗ ƙ ƴ, glottalised ʼy, digraphs |
| `hausa_tts/text.py` | normalisation, G2P, tone/length extraction, post-lexical rules, F0 targets |
| `hausa_tts/diacritizer.py` | lexicon / n-gram / neural tone-and-length restoration |
| `hausa_tts/modules.py` | text encoder, flows, WN, speaker encoder, prosody/pitch/energy heads |
| `hausa_tts/vocoder.py` | pitch-hint-conditioned HiFi-GAN + tonal discriminator |
| `hausa_tts/alignment.py` | MAS + monotonic path from durations |
| `hausa_tts/losses.py` | mel/adv/fm/kl + pitch, tone-classifier, harmonic, tonal-adversarial |
| `hausa_tts/model.py` | `HausaVITS` training forward + `infer()` |
| `hausa_tts/train.py` | adversarial trainer, checkpointing |
| `hausa_tts/inference.py` | synthesis API |
| `hausa_tts/metrics.py` | MCD, F0 RMSE/correlation, tone accuracy, RTF |

## Quick start
```bash
pip install torch numpy scipy
python tools/smoke_test.py            # 21-check end-to-end self-test on dummy data
python -m hausa_tts.train --config configs/single_speaker.yaml
```

## Data
BibleTTS Hausa (single speaker), Mozilla Common Voice Hausa and NaijaVoices
provide the raw material. Tone labels for the tone restorer come from tone-marked
lexica and from forced alignment plus F0 extraction on the audio (the self-supervised
route described in the plan).
