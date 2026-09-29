"""Hausa Tonal VITS: End-to-End Smoke Training on Kaggle/Colab with HuggingFace.

Dataset: suleiman2003/W_hausa_v7
Experiments:
  1. Single-Speaker Smoke Train (speaker_id: 88, 100 samples)
  2. Multi-Speaker Smoke Train (speaker_ids: 247 & 79, 50 samples each = 100 samples)

This script is self-contained. It can be run directly from the command line:
    python kaggle_smoke_train.py
Or pasted into a single cell of a Kaggle Notebook / Google Colab.
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
from pathlib import Path


# ==============================================================================
# 1. Environment & Package Installation (for Kaggle / Colab / Standalone)
# ==============================================================================
def ensure_dependencies():
    """Ensure all required libraries and the framework from GitHub are installed."""
    print("=" * 70)
    print("Step 1: Checking and installing dependencies...")
    print("=" * 70)

    required_pkgs = ["datasets", "soundfile", "scipy", "pyyaml"]
    for pkg in required_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing missing package: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Check if hausa_tts is already installed or available locally
    try:
        import hausa_tts
        print(f"✓ hausa_tts is available at: {hausa_tts.__file__}")
    except ImportError:
        # Check if running inside the repo folder
        if os.path.exists("hausa_tts_framework/hausa_tts"):
            sys.path.insert(0, "hausa_tts_framework")
            import hausa_tts
            print(f"✓ hausa_tts imported from local directory: {hausa_tts.__file__}")
        else:
            print("Installing hausa-tonal-vits directly from GitHub...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/issuleiman/hausa-tonal-vits.git"
            ])
            import hausa_tts
            print(f"✓ hausa_tts installed from GitHub: {hausa_tts.__file__}")

ensure_dependencies()


# Standard imports after dependencies are guaranteed
import time
import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset

from hausa_tts import (
    HausaVITSConfig,
    TrainConfig,
    HausaTTS,
    SynthOptions,
    save_wav,
    normalise,
)
from hausa_tts.train import HausaTTSTrainer
from hausa_tts.data import _guess_utt, load_wav, resample_linear
from hausa_tts.metrics import evaluate_pair, real_time_factor


# Detect environment (Kaggle / Colab / Terminal)
IN_NOTEBOOK = "ipykernel" in sys.modules or "google.colab" in sys.modules
if IN_NOTEBOOK:
    try:
        from IPython.display import Audio, display
    except ImportError:
        IN_NOTEBOOK = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 22050


# ==============================================================================
# 2. Dataset Streamer & Extractor (HuggingFace -> Local WAVs)
# ==============================================================================
def download_and_prepare_data(
    dataset_name: str = "suleiman2003/W_hausa_v7",
    single_spk: int = 88,
    multi_spks: tuple = (79, 247),
    target_count_single: int = 100,
    target_count_multi_per_spk: int = 50,
):
    """Streams samples from HuggingFace dataset and prepares local train/val filelists."""
    print("\n" + "=" * 70)
    print("Step 2: Streaming dataset from HuggingFace...")
    print(f"Dataset: {dataset_name}")
    print(f"Single Speaker ID: {single_spk} (target: {target_count_single} samples)")
    print(f"Multi Speaker IDs: {multi_spks} (target: {target_count_multi_per_spk} per speaker)")
    print("=" * 70)

    # Directories
    single_dir = Path("data_single_speaker")
    multi_dir = Path("data_multi_speaker")
    (single_dir / "wavs").mkdir(parents=True, exist_ok=True)
    (multi_dir / "wavs").mkdir(parents=True, exist_ok=True)

    print("Connecting to HuggingFace dataset stream (split='train')...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    collected_single = []
    collected_multi = {spk: [] for spk in multi_spks}

    total_target_single = target_count_single
    total_target_multi = len(multi_spks) * target_count_multi_per_spk

    processed = 0
    t0 = time.time()

    for item in ds:
        processed += 1
        spk_id = int(item.get("speaker_id", -1))
        text = item.get("text", "").strip()
        audio_data = item.get("audio", {})

        if not text or not audio_data:
            continue

        clean_text = normalise(text)
        if not clean_text or len(clean_text) < 3:
            continue

        # Extract waveform array and sample rate
        arr = audio_data.get("array")
        sr = audio_data.get("sampling_rate", TARGET_SR)

        if arr is None:
            continue

        arr = np.array(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Resample to 22050 Hz if needed
        if sr != TARGET_SR:
            arr = resample_linear(arr, sr, TARGET_SR)

        dur = len(arr) / float(TARGET_SR)
        if dur < 0.6 or dur > 12.0:
            continue

        utt_type = _guess_utt(clean_text)

        # Check single-speaker target
        if spk_id == single_spk and len(collected_single) < total_target_single:
            idx = len(collected_single)
            wav_path = single_dir / "wavs" / f"spk{spk_id}_{idx:04d}.wav"
            sf.write(str(wav_path), arr, TARGET_SR)
            # Format: wav_path|text|utt_type|durations|speaker_id
            collected_single.append(f"{wav_path.as_posix()}|{clean_text}|{utt_type}||0")
            if len(collected_single) % 25 == 0 or len(collected_single) == total_target_single:
                print(f"  [Single Speaker {single_spk}] Collected {len(collected_single)}/{total_target_single} samples")

        # Check multi-speaker target
        if spk_id in collected_multi and len(collected_multi[spk_id]) < target_count_multi_per_spk:
            # Map original speaker ID to contiguous integer ID (0, 1, ...)
            mapped_sid = multi_spks.index(spk_id)
            idx = len(collected_multi[spk_id])
            wav_path = multi_dir / "wavs" / f"spk{spk_id}_{idx:04d}.wav"
            sf.write(str(wav_path), arr, TARGET_SR)
            collected_multi[spk_id].append(f"{wav_path.as_posix()}|{clean_text}|{utt_type}||{mapped_sid}")
            if len(collected_multi[spk_id]) == target_count_multi_per_spk:
                print(f"  [Multi Speaker {spk_id}] Collected {len(collected_multi[spk_id])}/{target_count_multi_per_spk} samples")

        # Termination condition
        single_done = len(collected_single) >= total_target_single
        multi_done = all(len(v) >= target_count_multi_per_spk for v in collected_multi.values())
        if single_done and multi_done:
            break

        if processed % 500 == 0:
            print(f"  Scanned {processed} items ({time.time() - t0:.1f}s)...")

    print(f"Dataset streaming finished in {time.time() - t0:.1f}s. Scanned {processed} records.")

    # Write filelists: 90 train / 10 val splits
    def write_splits(items: list, out_dir: Path):
        np.random.seed(42)
        shuffled = items.copy()
        np.random.shuffle(shuffled)
        n_val = max(5, int(len(shuffled) * 0.10))
        val_set = shuffled[:n_val]
        train_set = shuffled[n_val:]
        with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(train_set) + "\n")
        with open(out_dir / "val.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(val_set) + "\n")
        print(f"  -> {out_dir}: {len(train_set)} train lines, {len(val_set)} val lines")
        return train_set, val_set

    print("\nSaving filelists...")
    write_splits(collected_single, single_dir)
    all_multi = []
    for spk_list in collected_multi.values():
        all_multi.extend(spk_list)
    write_splits(all_multi, multi_dir)


# ==============================================================================
# 3. Experiment Runner & Evaluator
# ==============================================================================
def run_smoke_training(
    exp_name: str,
    data_dir: str,
    is_multi_speaker: bool = False,
    n_speakers: int = 1,
    max_steps: int = 25,
    batch_size: int = 4,
):
    """Executes a smoke training experiment, saves checkpoint, and evaluates."""
    print("\n" + "=" * 70)
    print(f"Step 3: Starting Smoke Experiment: {exp_name.upper()}")
    print(f"Device: {DEVICE.upper()} | Max Steps: {max_steps} | Batch Size: {batch_size}")
    print("=" * 70)

    out_dir = Path("runs") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model Configuration
    if is_multi_speaker:
        cfg = HausaVITSConfig.multi_speaker(
            n_speakers=n_speakers,
            batch_size=batch_size,
            device=DEVICE,
            epochs=10,
        )
    else:
        cfg = HausaVITSConfig.single_speaker(
            batch_size=batch_size,
            device=DEVICE,
            epochs=10,
        )

    # 2. Training Configuration
    tcfg = TrainConfig(
        data_dir=data_dir,
        train_list=f"{data_dir}/train.txt",
        val_list=f"{data_dir}/val.txt",
        out_dir=str(out_dir),
        log_interval=5,
        save_interval=max_steps,
        fp16=(DEVICE == "cuda"),
    )

    # 3. Initialize Trainer
    trainer = HausaTTSTrainer(cfg, tcfg)

    print("\nRunning smoke training loop...")
    t0 = time.time()
    trainer.train(max_steps=max_steps)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s ({train_time/max_steps:.2f}s/step)")

    # 4. Checkpoint path
    ckpt_path = out_dir / f"G_final_{trainer.step}.pth"
    if not ckpt_path.exists():
        # Fallback to step checkpoint
        ckpt_path = out_dir / f"G_step_{trainer.step}.pth"

    print(f"Checkpoint saved at: {ckpt_path}")

    # 5. Validation Evaluation
    print("\nEvaluating on validation set...")
    val_items = []
    with open(f"{data_dir}/val.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("|")
                val_items.append((parts[0], parts[1], int(parts[4]) if len(parts) > 4 else 0))

    # Load trained model for inference
    spk_map = {f"spk_{i}": i for i in range(n_speakers)}
    tts = HausaTTS.from_pretrained(str(ckpt_path), device=DEVICE, speakers=spk_map)

    mcd_list = []
    f0_rmse_list = []
    rtf_list = []

    # Evaluate up to 3 validation samples
    eval_count = min(3, len(val_items))
    for i in range(eval_count):
        wav_path, text, sid = val_items[i]
        real_wav = load_wav(wav_path, TARGET_SR)

        # Synthesize with timing
        t_start = time.time()
        syn_wav = tts.speak(text, opts=SynthOptions(speaker=f"spk_{sid}"))
        gen_time = time.time() - t_start

        rtf = real_time_factor(gen_time, len(syn_wav) / float(TARGET_SR))
        rtf_list.append(rtf)

        # Compute pair metrics
        try:
            m = evaluate_pair(real_wav, syn_wav, TARGET_SR)
            if not np.isnan(m["mcd_db"]):
                mcd_list.append(m["mcd_db"])
            if not np.isnan(m["f0_rmse"]):
                f0_rmse_list.append(m["f0_rmse"])
        except Exception:
            pass

    avg_mcd = np.mean(mcd_list) if mcd_list else 0.0
    avg_f0 = np.mean(f0_rmse_list) if f0_rmse_list else 0.0
    avg_rtf = np.mean(rtf_list) if rtf_list else 0.0

    print("-" * 50)
    print(f"Validation Results for [{exp_name}]:")
    print(f"  • Real-Time Factor (RTF) : {avg_rtf:.4f} (lower is faster)")
    print(f"  • Avg Mel Cepstral Dist  : {avg_mcd:.2f} dB (MCD)")
    print(f"  • Avg F0 RMSE            : {avg_f0:.4f}")
    print("-" * 50)

    return tts, ckpt_path


# ==============================================================================
# 4. Speech Synthesis Demonstration
# ==============================================================================
def demonstrate_synthesis(tts: HausaTTS, exp_type: str, test_sentences: list, speakers: list):
    """Synthesizes sample sentences and saves WAV files."""
    print(f"\nDemonstrating synthesis for {exp_type}...")
    output_dir = Path("synthesized_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for sid_name in speakers:
        for idx, sentence in enumerate(test_sentences):
            audio = tts.speak(
                sentence,
                opts=SynthOptions(
                    speaker=sid_name,
                    rate=1.0,
                    pitch=0.0,
                    pitch_range=1.0,
                ),
            )
            out_path = output_dir / f"{exp_type}_{sid_name}_sample{idx+1}.wav"
            save_wav(audio, str(out_path), sample_rate=TARGET_SR)
            print(f"  ✓ [{sid_name}] \"{sentence}\" -> {out_path}")

            if IN_NOTEBOOK:
                print(f"Audio Playback ({sid_name}):")
                display(Audio(audio, rate=TARGET_SR))


# ==============================================================================
# 5. Main Execution Pipeline
# ==============================================================================
def main():
    print("=" * 70)
    print("   HAUSA TONAL VITS: SMOKE EXPERIMENT PIPELINE")
    print("   Single-Speaker (88) & Multi-Speaker (79, 247)")
    print("=" * 70)

    # 1. Download & Prepare Datasets (100 samples each)
    download_and_prepare_data(
        dataset_name="suleiman2003/W_hausa_v7",
        single_spk=88,
        multi_spks=(79, 247),
        target_count_single=100,
        target_count_multi_per_spk=50,
    )

    test_sentences = [
        "Yaushe za ka dawo gida?",                     # Wh-question with tone
        "Barka da yamma, ya aiki da gajiya?",          # Statement greeting
        "Shin za ku iya tafiya tare da mu yau?",       # Yes/No question
    ]

    # 2. Experiment 1: Single Speaker (speaker 88)
    tts_single, _ = run_smoke_training(
        exp_name="single_speaker_88",
        data_dir="data_single_speaker",
        is_multi_speaker=False,
        n_speakers=1,
        max_steps=25,
        batch_size=4,
    )
    demonstrate_synthesis(
        tts=tts_single,
        exp_type="single_spk88",
        test_sentences=test_sentences[:2],
        speakers=["spk_0"],
    )

    # 3. Experiment 2: Multi Speaker (speakers 79 & 247)
    tts_multi, _ = run_smoke_training(
        exp_name="multi_speaker_79_247",
        data_dir="data_multi_speaker",
        is_multi_speaker=True,
        n_speakers=2,
        max_steps=25,
        batch_size=4,
    )
    demonstrate_synthesis(
        tts=tts_multi,
        exp_type="multi_spk",
        test_sentences=test_sentences[:2],
        speakers=["spk_0", "spk_1"],   # spk_0 = 79, spk_1 = 247
    )

    print("\n" + "=" * 70)
    print("✅ All smoke experiments completed successfully!")
    print("Check the 'runs/' directory for model checkpoints and")
    print("'synthesized_samples/' for generated WAV audio files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
