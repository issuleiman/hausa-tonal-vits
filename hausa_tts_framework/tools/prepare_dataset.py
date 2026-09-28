"""Dataset preparation and validation utility for Hausa Tonal VITS.

Validates audio files, applies text normalisation, computes audio durations,
and generates train/val filelists in the expected pipe-delimited format:
    wav_path|text|utt_type|durations|speaker_id
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import soundfile as sf
except ImportError:
    sf = None

# Ensure hausa_tts is importable
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from hausa_tts.text import normalise
from hausa_tts.data import _guess_utt


def get_audio_info(wav_path: str) -> Tuple[float, int]:
    """Return duration in seconds and sample rate."""
    if sf is not None:
        try:
            info = sf.info(wav_path)
            return float(info.duration), int(info.samplerate)
        except Exception:
            pass
    # Fallback to wave module
    import wave
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        dur = frames / float(sr)
        return dur, sr


def process_dataset(
    input_file: str,
    output_dir: str,
    val_ratio: float = 0.05,
    min_duration: float = 0.5,
    max_duration: float = 12.0,
    target_sr: int = 22050,
    speaker_id: str = "0",
    seed: int = 1234,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading input metadata from: {input_file}")
    valid_entries: List[str] = []
    skipped_not_found = 0
    skipped_duration = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        wav_path = parts[0].strip()
        raw_text = parts[1].strip()

        # Optional extra fields
        utt_type = parts[2].strip() if len(parts) > 2 and parts[2].strip() else _guess_utt(raw_text)
        durations = parts[3].strip() if len(parts) > 3 else ""
        spk = parts[4].strip() if len(parts) > 4 and parts[4].strip() else speaker_id

        if not os.path.exists(wav_path):
            skipped_not_found += 1
            continue

        try:
            dur, sr = get_audio_info(wav_path)
        except Exception as e:
            print(f"Warning: could not read {wav_path}: {e}")
            skipped_not_found += 1
            continue

        if dur < min_duration or dur > max_duration:
            skipped_duration += 1
            continue

        clean_text = normalise(raw_text)
        if not clean_text:
            continue

        entry = f"{wav_path}|{clean_text}|{utt_type}|{durations}|{spk}"
        valid_entries.append(entry)

    total_valid = len(valid_entries)
    print(f"Total entries processed: {len(lines)}")
    print(f"Valid entries: {total_valid}")
    print(f"Skipped (missing file): {skipped_not_found}")
    print(f"Skipped (duration < {min_duration}s or > {max_duration}s): {skipped_duration}")

    if total_valid == 0:
        print("Error: No valid entries found.")
        return

    random.shuffle(valid_entries)
    n_val = max(1, int(total_valid * val_ratio))
    val_entries = sorted(valid_entries[:n_val])
    train_entries = sorted(valid_entries[n_val:])

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_entries) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_entries) + "\n")

    print(f"Saved {len(train_entries)} training lines -> {train_path}")
    print(f"Saved {len(val_entries)} validation lines -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Hausa Tonal VITS")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input file with 'wav_path|text' per line")
    parser.add_argument("--output_dir", "-o", type=str, default="data",
                        help="Directory to save train.txt and val.txt")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of data for validation (default 0.05)")
    parser.add_argument("--min_duration", type=float, default=0.5,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=12.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--speaker_id", type=str, default="0",
                        help="Default speaker ID (default '0')")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        speaker_id=args.speaker_id,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
