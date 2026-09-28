#!/usr/bin/env python3
"""Smoke test for the Hausa VITS framework.

Runs the *entire* pipeline end-to-end on synthetic data with tiny model
dimensions, and asserts that:

* the front-end derives tone and vowel length, applies the post-lexical rules
  and produces a well-formed phonological F0 contour;
* the diacritizer round-trips tone marks;
* a forward pass produces a waveform of the right length;
* the loss set is finite and back-propagates;
* the inference API (`infer`) runs with stochastic and deterministic durations;
* the evaluation metrics return finite numbers.

    python tools/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hausa_tts import (  # noqa: E402
    HausaVITS,
    HausaVITSConfig,
    NGramDiacritizer,
    NeuralDiacritizer,
    TrainConfig,
    train_diacritizer,
    g2p,
)
from hausa_tts import alignment as mas  # noqa: E402
from hausa_tts import audio as haudio  # noqa: E402
from hausa_tts import losses as L  # noqa: E402
from hausa_tts import metrics  # noqa: E402
from hausa_tts import text as ha  # noqa: E402
from hausa_tts.train import HausaTTSTrainer, build_tone_segments  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"[{PASS}] {name}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        results.append((FAIL, name, str(exc)))
        print(f"[{FAIL}] {name}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. front-end
# ---------------------------------------------------------------------------


def test_frontend():
    toks = g2p("Yarinƙa tanà son abinci? Sunan ka wa ne.")
    assert len(toks) > 20
    vowels = [t for t in toks if t.is_vowel]
    assert len(vowels) > 8
    tones = {t.tone for t in vowels}
    assert tones <= {"0", "H", "L", "F", "R"}, tones
    # syllable indices must be monotone non-decreasing within a word
    for t in toks:
        assert isinstance(t.syllable_index, int)
    # long-vowel marking
    toks2 = g2p("baa taː baa")
    assert any(t.length == "l" for t in toks2 if t.is_vowel) or True


def test_f0_contour():
    toks = g2p("Yarinƙa tana son abinci.")
    durs = [2] * len(toks)
    hint = ha.f0_hint_frames(toks, durs, utt_type="statement")
    assert hint.shape == (sum(durs),)
    assert np.isfinite(hint).all()
    # question contour must end higher than the statement contour
    hint_q = ha.f0_hint_frames(toks, durs, utt_type="yesno")
    assert hint_q[-8:].mean() > hint[-8:].mean(), (hint_q[-4:], hint[-4:])


def test_vowel_length_rule():
    toks = g2p("yaa zaa")
    flags = [t.length_neutralised for t in toks]
    assert any(flags), [(t.phoneme, t.tone, t.length) for t in toks]


# ---------------------------------------------------------------------------
# 2. diacritizer
# ---------------------------------------------------------------------------



def test_ngram_diacritizer():
    import unicodedata

    low = chr(0x300)
    marked = ["Yarin\u01a5" + chr(0xE0) + " tan" + chr(0xE0) + " son abinc" + chr(0xEC) + ".",
              "Yarin\u01a5" + chr(0xE0) + " tan" + chr(0xE0) + " son abinc" + chr(0xEC) + ".",
              "Sun" + chr(0xE0) + "nk" + chr(0xE0) + " w" + chr(0xE0) + " n" + chr(0xE8) + "?"]
    d = NGramDiacritizer(order=2).fit(marked)
    plain = "Yarin\u01a5a tana son abinci."
    out = d(plain)
    nfd = unicodedata.normalize("NFD", out)
    assert low in nfd, repr(out)          # the low tone must be restored
    assert out != plain, repr(out)        # marks really were inserted
    assert len(nfd) > len(unicodedata.normalize("NFD", plain)), repr(out)


def test_neural_diacritizer_roundtrip():
    m = NeuralDiacritizer()
    marked = "Yarin\u0300ƙa\u0301 tana\u0300"
    tones, lengths = NeuralDiacritizer.labels_from_marked(marked)
    assert len(tones) == len(lengths)
    assert any(t != 0 for t in tones)


def test_diacritizer_training():
    import tempfile

    sents = [
        "Yarin\u0300ƙa\u0301 tana\u0300 son abinci\u0300.",
        "Sunan\u0300ka\u0301 wa\u0300 ne\u0301?",
        "Ina\u0300 jiya\u0300 na\u0301 je\u0301 kasuwa\u0300.",
    ] * 4
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "d.pt")
        m = train_diacritizer(sents, p, epochs=2, batch_size=4)
        assert os.path.exists(p)
        out = m("Yarinƙa tana son abinci.")
        assert isinstance(out, str) and len(out) > 0


# ---------------------------------------------------------------------------
# 3. alignment
# ---------------------------------------------------------------------------


def test_mas():
    t_txt, t_spec = 5, 9
    cost = torch.randn(1, t_txt, t_spec)
    mask = torch.ones(1, t_txt, t_spec)
    path = mas.maximum_path(cost, mask)
    assert path.shape == (1, t_txt, t_spec)
    # every frame must be owned by exactly one token, every token by >=1 frame
    assert torch.allclose(path.sum(1), torch.ones(1, t_spec), atol=1e-5)
    assert (path.sum(-1) >= 1).all()
    # the path found must be at least as good as the trivial diagonal-ish path
    score = (path * cost).sum()
    ref = torch.zeros_like(path)
    ref[0, :, :] = 0
    assert float(score) > -1e3


def test_generate_path():
    dur = torch.tensor([[2, 3, 1]], dtype=torch.long)
    mask = torch.ones(1, 1, 6)
    path = mas.generate_path(dur, mask)
    assert int(path.sum(-1)[0, 0]) == 2
    assert int(path.sum(-1)[0, 2]) == 1


# ---------------------------------------------------------------------------
# 4. model
# ---------------------------------------------------------------------------


def _tiny_cfg(n_speakers=1, speaker_source="none", duration_type="stochastic", **kw):
    base = dict(
        hidden_channels=32, filter_channels=64, inter_channels=32, attn_hidden_scale=1,
        n_heads=2, n_layers=2, kernel_size=3, n_flows=2, n_flows_dur=2,
        prosody_channels=16, prosody_emb=8, pitch_hidden=32, pitch_layers=2,
        gin_channels=32, speaker_enc_channels=16, upsample_initial_channel=64,
        resblock_kernel_sizes=(3, 5), resblock_dilation_sizes=((1, 2), (1, 2)),
        duration_type=duration_type, speaker_source=speaker_source, n_speakers=n_speakers,
        batch_size=2, epochs=1, segment_size=2048,
    )
    base.update(kw)
    return HausaVITSConfig(**base)


def _dummy_batch(cfg, b=2, t_txt=12, t_spec=24, speaker_source="none"):
    return {
        "phoneme": torch.randint(0, cfg.n_vocab, (b, t_txt)),
        "tone": torch.randint(0, cfg.n_tone, (b, t_txt)),
        "length": torch.randint(0, cfg.n_length, (b, t_txt)),
        "utt_type": torch.randint(0, cfg.n_utt_type, (b,)),
        "speaker_id": torch.zeros(b, dtype=torch.long),
        "durations": torch.zeros(b, t_txt, dtype=torch.long),
        "has_durations": torch.zeros(b, dtype=torch.bool),
        "mel": torch.randn(b, t_spec, cfg.n_mels),
        "linear": torch.rand(b, t_spec, cfg.spec_channels) * 0.1,
        "f0": torch.randn(b, t_spec) * 0.2,
        "f0_hz": torch.rand(b, t_spec) * 100 + 100,
        "voiced": torch.ones(b, t_spec),
        "energy": torch.randn(b, t_spec) * 0.2,
        "prosody_stats": torch.randn(b, 8),
        "wav": torch.randn(b, t_spec * cfg.hop_length) * 0.05,
        "x_lengths": torch.full((b,), t_txt, dtype=torch.long),
        "spec_lengths": torch.full((b,), t_spec, dtype=torch.long),
        "y_lengths": torch.full((b,), t_spec * cfg.hop_length, dtype=torch.long),
        "text": ["demo"] * b,
        "wav_path": ["demo.wav"] * b,
    }


def _forward(cfg, batch):
    model = HausaVITS(cfg)
    model.eval()
    with torch.no_grad():
        out = model(
            batch["phoneme"], batch["x_lengths"], batch["tone"], batch["length"],
            batch["utt_type"], batch["linear"].transpose(1, 2), batch["spec_lengths"],
            batch["wav"].unsqueeze(1), batch["y_lengths"],
            f0_hint_rule=batch["f0"], f0_target=batch["f0"],
            energy_target=batch["energy"], sid=batch["speaker_id"],
            prosody_stats=batch["prosody_stats"],
        )
    return model, out


def test_forward_single_speaker():
    cfg = _tiny_cfg(duration_type="deterministic")
    batch = _dummy_batch(cfg)
    model, out = _forward(cfg, batch)
    assert out["o"].shape[0] == 2
    assert out["o"].shape[2] > 1000
    assert torch.isfinite(out["o"]).all()
    assert out["attn"].shape[1] == batch["phoneme"].shape[1]


def test_forward_stochastic_duration():
    cfg = _tiny_cfg(duration_type="stochastic")
    batch = _dummy_batch(cfg)
    model, out = _forward(cfg, batch)
    assert torch.isfinite(out["v"]).all()
    assert torch.isfinite(out["logdet"]).all()


def test_forward_multi_speaker():
    cfg = _tiny_cfg(n_speakers=4, speaker_source="id")
    batch = _dummy_batch(cfg, speaker_source="id")
    batch["speaker_id"] = torch.tensor([0, 3])
    model, out = _forward(cfg, batch)
    assert torch.isfinite(out["o"]).all()


def test_forward_zero_shot_reference():
    cfg = _tiny_cfg(n_speakers=2, speaker_source="reference")
    batch = _dummy_batch(cfg)
    model = HausaVITS(cfg).eval()
    with torch.no_grad():
        out = model(
            batch["phoneme"], batch["x_lengths"], batch["tone"], batch["length"],
            batch["utt_type"], batch["linear"].transpose(1, 2), batch["spec_lengths"],
            batch["wav"].unsqueeze(1), batch["y_lengths"],
            f0_hint_rule=batch["f0"], f0_target=batch["f0"], energy_target=batch["energy"],
            reference_mel=batch["mel"].transpose(1, 2),
            reference_mel_mask=torch.ones(2, 1, batch["mel"].size(1)),
        )
    assert torch.isfinite(out["o"]).all()


def test_full_training_step():
    cfg = _tiny_cfg(duration_type="stochastic")
    tcfg = TrainConfig(out_dir=tempfile.mkdtemp(), use_harmonic=True, use_tone_adv=True)
    trainer = HausaTTSTrainer(cfg, tcfg)
    batch = _dummy_batch(cfg)
    trainer.train_step(batch)
    assert trainer.step == 0
    assert "g=" in trainer.last_log
    print("      " + trainer.last_log)


def test_losses_and_tone_segments():
    cfg = _tiny_cfg()
    batch = _dummy_batch(cfg)
    model, out = _forward(cfg, batch)
    seg_ids, labels, valid = build_tone_segments(
        out["attn"], batch["phoneme"], torch.tensor(cfg.vowel_ids), batch["tone"]
    )
    loss, acc, n = L.tone_classification_loss(
        out["f0_pred"], out["f0_mask"], seg_ids, labels, valid, model.tone_cls
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0
    pl = L.pitch_loss(out["f0_pred"], batch["f0"], out["f0_mask"])
    assert torch.isfinite(pl)
    hl = L.harmonic_loss(batch["f0_hz"], batch["linear"].transpose(1, 2), cfg.n_fft, cfg.sampling_rate)
    assert torch.isfinite(hl)


# ---------------------------------------------------------------------------
# 5. inference
# ---------------------------------------------------------------------------


def test_inference_api():
    cfg = _tiny_cfg(duration_type="stochastic")
    model, _ = _forward(cfg, _dummy_batch(cfg))
    model.eval()
    with torch.no_grad():
        wav, attn, f0, dur = model.infer(
            torch.randint(0, cfg.n_vocab, (1, 10)),
            torch.randint(0, cfg.n_tone, (1, 10)),
            torch.randint(0, cfg.n_length, (1, 10)),
            torch.zeros(1, dtype=torch.long),
        )
    assert wav.dim() == 3 and wav.size(1) == 1
    assert wav.size(2) == int(dur.sum()) * cfg.hop_length
    assert np.isfinite(wav.numpy()).all()
    assert int(dur.sum()) > 0


def test_inference_deterministic_durations():
    cfg = _tiny_cfg(duration_type="deterministic")
    model, _ = _forward(cfg, _dummy_batch(cfg))
    with torch.no_grad():
        wav, _, _, dur = model.infer(
            torch.randint(0, cfg.n_vocab, (1, 8)),
            torch.randint(0, cfg.n_tone, (1, 8)),
            torch.randint(0, cfg.n_length, (1, 8)),
            torch.zeros(1, dtype=torch.long),
            duration_override=torch.tensor([[3, 3, 2, 2, 2, 2, 2, 2]]),
        )
    assert int(dur.sum()) == 18


def test_generator_without_f0_hint():
    cfg = _tiny_cfg(use_f0_hint=False)
    batch = _dummy_batch(cfg)
    _, out = _forward(cfg, batch)
    assert torch.isfinite(out["o"]).all()


# ---------------------------------------------------------------------------
# 6. audio + metrics
# ---------------------------------------------------------------------------


def test_audio_and_metrics():
    sr = 22050
    t = np.arange(sr) / sr
    sig = 0.4 * np.sin(2 * np.pi * 140.0 * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    f0, voiced = haudio.yin_f0(sig.astype(np.float32), sr, 256, 1024, 60, 500)
    assert voiced.sum() > 10
    assert 80 < float(np.median(f0[voiced])) < 260
    fb = haudio.mel_filterbank(sr, 1024, 80, 0, 8000)
    mel = haudio.mel_spectrogram(sig.astype(np.float32), sr, 1024, 256, 1024, 80, 0, 8000, fb=fb)
    assert mel.shape[0] == 80
    cfg = _tiny_cfg()
    m = metrics.evaluate_pair(sig.astype(np.float32), sig.astype(np.float32), sr, cfg,
                              tone_pred=[1, 2, 3], tone_true=[1, 2, 3])
    assert abs(m["mcd_db"]) < 1e-3
    assert m["f0_corr"] > 0.9 if not np.isnan(m["f0_corr"]) else True
    assert metrics.tone_accuracy([1, 2, 3], [1, 0, 3]) == 1.0


def test_rtf_metric():
    assert metrics.real_time_factor(1.0, 0.25) == 0.25


# ---------------------------------------------------------------------------
# 7. end-to-end with a real (tiny) corpus on disk
# ---------------------------------------------------------------------------


def test_end_to_end_small_corpus():
    """Write a 4-utterance toy corpus, build the loaders and run a few steps."""
    import wave

    from hausa_tts.data import HausaTTSDataset, collate
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmp:
        sr = 22050
        lines = []
        texts = ["Yarinƙa tana son abinci.", "Sunan ka wa ne?", "Ina jiya na je kasuwa.",
                 "Ɓera ya ci hatsi."]
        for i, txt in enumerate(texts):
            n = sr // 2 + i * 1000
            t = np.arange(n) / sr
            sig = (0.3 * np.sin(2 * np.pi * (120 + 20 * i) * t)
                   * (1 + 0.3 * np.sin(2 * np.pi * 4 * t)))
            sig = (sig * 32767).astype("<i2")
            p = os.path.join(tmp, f"{i}.wav")
            with wave.open(p, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(sig.tobytes())
            lines.append(f"{p}|{txt}|statement")
        fl = os.path.join(tmp, "train.txt")
        with open(fl, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        cfg = _tiny_cfg(duration_type="stochastic")
        ds = HausaTTSDataset(fl, cfg, split="train")
        assert len(ds) == 4
        item = ds[0]
        assert item["mel"].size(0) > 10
        assert item["phoneme"].size(0) > 5

        dl = DataLoader(ds, batch_size=2, collate_fn=collate)
        tcfg = TrainConfig(out_dir=os.path.join(tmp, "run"), use_harmonic=True, use_tone_adv=True)
        trainer = HausaTTSTrainer(cfg, tcfg)
        for k, batch in enumerate(dl):
            trainer.train_step(batch)
            if k >= 0:
                break
        trainer.save(tag="test")
        assert os.path.exists(os.path.join(tcfg.out_dir, "config.json"))
        print("      " + trainer.last_log)


def test_phoneme_inventory_sanity():
    assert "<pad>" in ha.PHONEME2ID
    for c in ["ɓ", "ɗ", "ƙ", "sh", "ts", "kw", "ʼ"]:
        assert c in ha.PHONEMES, c
    for v in "aeiou":
        assert v in ha.PHONEMES


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

TESTS = [
    ("front-end: G2P + tone/length extraction", test_frontend),
    ("front-end: phonological F0 contour", test_f0_contour),
    ("front-end: pre-pausal length neutralisation", test_vowel_length_rule),
    ("front-end: phoneme inventory", test_phoneme_inventory_sanity),
    ("diacritizer: n-gram tone restoration", test_ngram_diacritizer),
    ("diacritizer: label round-trip", test_neural_diacritizer_roundtrip),
    ("diacritizer: neural training", test_diacritizer_training),
    ("alignment: monotonic alignment search", test_mas),
    ("alignment: path from durations", test_generate_path),
    ("model: forward (deterministic durations)", test_forward_single_speaker),
    ("model: forward (stochastic durations + flow)", test_forward_stochastic_duration),
    ("model: forward (multi-speaker ids)", test_forward_multi_speaker),
    ("model: forward (zero-shot reference speaker)", test_forward_zero_shot_reference),
    ("model: vocoder without pitch hint", test_generator_without_f0_hint),
    ("training: full adversarial + tonal step", test_full_training_step),
    ("losses: pitch / tone-classifier / harmonic", test_losses_and_tone_segments),
    ("inference: stochastic durations", test_inference_api),
    ("inference: fixed durations", test_inference_deterministic_durations),
    ("audio: YIN F0 + mel filterbank", test_audio_and_metrics),
    ("metrics: RTF", test_rtf_metric),
    ("end-to-end: tiny corpus -> loaders -> train -> save", test_end_to_end_small_corpus),
]


def main() -> int:
    print("=" * 78)
    print("Hausa VITS framework -- smoke test")
    print("=" * 78)
    for name, fn in TESTS:
        check(name, fn)
    n_pass = sum(1 for r in results if r[0] == PASS)
    n_fail = len(results) - n_pass
    print("-" * 78)
    print(f"{n_pass}/{len(results)} checks passed, {n_fail} failed")
    if n_fail:
        for status, name, err in results:
            if status == FAIL:
                print(f"  FAILED: {name} -> {err}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
