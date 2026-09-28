"""Tone and vowel-length restoration for Hausa (diacritization front-end).

**Why this module exists.** Standard Boko orthography does not mark tone, and
ordinary Hausa text does not mark vowel length, even though both are
phonemically contrastive: ``korá``/``kora`` type minimal pairs, and the
``bàː``/``ba`` length opposition, change word meaning entirely
(Newman & Van Heuven 1981; Newman 1997; r12a orthography notes). A TTS system
trained on unmarked text therefore sees an underspecified input.

The literature on the sister language Yorùbá shows this is a standard,
solvable sub-task -- tone-mark restoration is treated as a diacritic
restoration problem solved with attention-based seq2seq models, LSTMs, or
fine-tuned LLMs (Orife et al.; Asahiah et al. 2017; Toyin et al. 2025). This
module gives you three interchangeable strategies, so the acoustic model can
be trained even when no tone-annotated text exists:

1. ``LexiconDiacritizer`` -- exact lookup against a tone-marked Hausa lexicon
   (highest precision; build it from any annotated resource).
2. ``NgMramDiacritizer`` -- a word-level n-gram back-off with a rule-based
   fallback: unmarked words get the front-end's default reading (high tone,
   short vowel), which is exactly what the unmarked orthography implies.
3. ``NeuralDiacritizer`` -- a character-level BiLSTM tagger over
   (tone, length) labels, trained on whatever annotated text is available.
   Works cross-lingually for Hausa/Yorùbá/Igbo because the task is shallow.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TO_NFD_TONE = {"H": "\u0301", "L": "\u0300", "F": "\u0302", "R": "\u0300"}
TONE_LABELS = ["0", "H", "L", "F"]
LENGTH_LABELS = ["-", "s", "l"]


# ---------------------------------------------------------------------------
# 1. lexicon
# ---------------------------------------------------------------------------


class LexiconDiacritizer:
    """Exact word-level lookup, then delegate unknown words to a fallback."""

    def __init__(self, lexicon: Dict[str, str], fallback=None) -> None:
        self.lexicon = {self._key(k): v for k, v in lexicon.items()}
        self.fallback = fallback

    @staticmethod
    def _key(word: str) -> str:
        import unicodedata

        nfd = unicodedata.normalize("NFD", word.lower())
        return "".join(c for c in nfd if not unicodedata.combining(c))

    @classmethod
    def from_file(cls, path: str, fallback=None) -> "LexiconDiacritizer":
        with open(path, "r", encoding="utf-8") as fh:
            pairs = [line.strip().split("\t") for line in fh if line.strip()]
        lex = {}
        for p in pairs:
            if len(p) >= 2:
                lex[p[0]] = p[1]
        return cls(lex, fallback=fallback)

    def __call__(self, text: str) -> str:
        out = []
        for tok in re.findall(r"\S+|\s+", text):
            if tok.isspace():
                out.append(tok)
                continue
            hit = self.lexicon.get(self._key(re.sub(r"[^\w\u0253\u0257\u0199\u02bc]", "", tok)))
            if hit:
                out.append(hit)
            elif self.fallback is not None:
                out.append(self.fallback(tok))
            else:
                out.append(tok)
        return "".join(out)


# ---------------------------------------------------------------------------
# 2. n-gram / rule back-off
# ---------------------------------------------------------------------------


class NGramDiacritizer:
    """Word n-gram tone/length back-off over the *unmarked* orthographic form.

    The table is keyed by ``(context words, current word)`` where every key is
    tone/length-stripped, and the value is the surface form observed in the
    annotated corpus. Stripping the key is essential: standard Boko text is
    written without tone or length marks, so an unmarked query can only match an
    unmarked key. Unigram back-off covers unseen contexts and the first word of
    a sentence. This captures the grammaticalised tonal alternations of Hausa
    (noun-class and genitive linker words, verbal grade alternations) with no
    neural training, which makes it a usable baseline for a small corpus.
    """

    def __init__(self, order: int = 2) -> None:
        self.order = order
        self._tables: List[Dict[Tuple[str, ...], Counter]] = [defaultdict(Counter) for _ in range(order)]
        self._unigram: Dict[str, Counter] = defaultdict(Counter)

    @staticmethod
    def _key(w: str) -> str:
        import unicodedata

        nfd = unicodedata.normalize("NFD", w.lower())
        return "".join(c for c in nfd if not unicodedata.combining(c))

    def fit(self, marked_sentences: Iterable[str]) -> "NGramDiacritizer":
        for sent in marked_sentences:
            words = sent.split()
            keys = [self._key(w) for w in words]
            for i, w in enumerate(words):
                self._unigram[keys[i]][w] += 1
                for o in range(1, self.order + 1):
                    ctx = tuple(keys[max(0, i - o) : i]) + (keys[i],)
                    self._tables[o - 1][ctx][w] += 1
        return self

    def _predict(self, ctx_keys: Sequence[str], current: str) -> Optional[str]:
        for o in range(min(self.order, len(ctx_keys)), 0, -1):
            ctx = tuple(ctx_keys[-o:]) + (current,)
            counter = self._tables[o - 1].get(ctx)
            if counter:
                return counter.most_common(1)[0][0]
        uni = self._unigram.get(current)
        return uni.most_common(1)[0][0] if uni else None

    def __call__(self, text: str) -> str:
        words = text.split()
        keys = [self._key(w) for w in words]
        out = []
        for i, w in enumerate(words):
            pred = self._predict(keys[:i], keys[i])
            out.append(pred if pred is not None else w)
        return " ".join(out)

    def save(self, path: str) -> None:
        blob = {
            "order": self.order,
            "tables": [{"\u0001".join(k): c.most_common(1)[0][0] for k, c in t.items()} for t in self._tables],
            "unigram": {k: c.most_common(1)[0][0] for k, c in self._unigram.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "NGramDiacritizer":
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        obj = cls(order=blob["order"])
        for o, table in enumerate(blob["tables"]):
            for k, v in table.items():
                obj._tables[o][tuple(k.split("\u0001"))][v] = 1
        for k, v in blob.get("unigram", {}).items():
            obj._unigram[k][v] = 1
        return obj


# ---------------------------------------------------------------------------
# 3. neural tagger
# ---------------------------------------------------------------------------


class NeuralDiacritizer(nn.Module):
    """Character-level BiLSTM tagger predicting tone and vowel length jointly.

    Input: unmarked characters. Output: for every character position, a tone
    class (none / H / L / F) and a length class (none / short / long). Trained
    with cross-entropy on annotated sentences. At inference, the predictions are
    written back onto the text as combining diacritics, which is exactly the
    input :func:`hausa_tts.text.g2p` expects.

    Deliberately shallow: the task is a tagging problem, and a 2-layer BiLSTM
    reaches usable accuracy with a few thousand annotated Hausa sentences, in
    line with the diacritic-restoration literature for Yorùbá and Arabic.
    """

    def __init__(self, n_chars: int = 256, emb: int = 64, hidden: int = 256,
                 n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb)
        self.lstm = nn.LSTM(emb * 3, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.tone_head = nn.Linear(hidden * 2, len(TONE_LABELS))
        self.length_head = nn.Linear(hidden * 2, len(LENGTH_LABELS))
        self.char2id = {chr(i): i for i in range(n_chars)}
        self.id2char = {i: chr(i) for i in range(n_chars)}

    def _features(self, ids: torch.Tensor) -> torch.Tensor:
        base = self.emb(ids)
        # simple shape features: is vowel / is consonant / is space
        vowels = torch.tensor(
            [ord(c) for c in "aeiou"], device=ids.device
        )
        is_vowel = torch.isin(ids, vowels).float().unsqueeze(-1)
        is_space = (ids == ord(" ")).float().unsqueeze(-1)
        return torch.cat([base, is_vowel.expand(-1, -1, base.size(-1)),
                          is_space.expand(-1, -1, base.size(-1))], dim=-1)

    def forward(self, ids: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                self._features(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                      total_length=ids.size(1))
        else:
            out, _ = self.lstm(self._features(ids))
        return self.tone_head(out), self.length_head(out)

    # -- text <-> ids ---------------------------------------------------

    @staticmethod
    def iter_chars(text: str):
        """Yield ``(base_char, combining_marks)`` -- one item per base char."""
        import unicodedata

        decomp = unicodedata.normalize("NFD", text.lower())
        out = []
        i = 0
        while i < len(decomp):
            ch = decomp[i]
            j = i + 1
            mods = []
            while j < len(decomp) and unicodedata.combining(decomp[j]):
                mods.append(decomp[j]); j += 1
            out.append((ch, "".join(mods)))
            i = j
        return out

    def encode_text(self, text: str):
        chars = self.iter_chars(text)
        return torch.tensor([[self.char2id.get(c, 0) for c, _ in chars]], dtype=torch.long)

    def decode_to_marked(self, text: str, tone_ids: torch.Tensor,
                         length_ids: torch.Tensor) -> str:
        import unicodedata

        marked = []
        for i, (ch, _) in enumerate(self.iter_chars(text)):
            if ch not in "aeiou" or i >= tone_ids.size(0):
                marked.append(ch)
                continue
            add = ""
            if length_ids[i].item() == 2:
                add += "\u0304"
            add += TO_NFD_TONE.get(TONE_LABELS[int(tone_ids[i].item())], "")
            if tone_ids[i].item() == 2 and not add.endswith("\u0300"):
                pass
            marked.append(ch + add)
        return unicodedata.normalize("NFC", "".join(marked))

    @torch.no_grad()
    def __call__(self, text: str) -> str:
        self.eval()
        ids = self.encode_text(text)
        tone_logits, len_logits = self.forward(ids)
        return self.decode_to_marked(text, tone_logits[0].argmax(-1), len_logits[0].argmax(-1))

    # -- training helpers ------------------------------------------------

    @staticmethod
    def labels_from_marked(text: str):
        """Read ``(tone, length)`` labels off a tone-marked string, one per base char."""
        import unicodedata

        tones, lengths = [], []
        for ch, mods in NeuralDiacritizer.iter_chars(text):
            tone, length = "0", "-"
            for v in mods:
                if v == "\u0301":
                    tone = "H"
                elif v in ("\u0300", "\u030f"):
                    tone = "L"
                elif v in ("\u0302", "\u030c"):
                    tone = "F"
                elif v == "\u0304":
                    length = "l"
                elif v == "\u0306":
                    length = "s"
            if ch in "aeiou":
                if length == "-":
                    length = "s"
                tones.append(TONE_LABELS.index(tone))
                lengths.append(LENGTH_LABELS.index(length))
            else:
                tones.append(0)
                lengths.append(0)
        return tones, lengths


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------


def build_diacritizer(lexicon_path: Optional[str] = None,
                      ngram_path: Optional[str] = None,
                      neural_ckpt: Optional[str] = None):
    """Return the best available diacritizer, or ``None`` for the default reading."""
    if neural_ckpt and os.path.exists(neural_ckpt):
        model = NeuralDiacritizer()
        state = torch.load(neural_ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        return model
    if lexicon_path and os.path.exists(lexicon_path):
        base = NGramDiacritizer.load(ngram_path) if ngram_path and os.path.exists(ngram_path) else None
        return LexiconDiacritizer.from_file(lexicon_path, fallback=base)
    if ngram_path and os.path.exists(ngram_path):
        return NGramDiacritizer.load(ngram_path)
    return None


def train_diacritizer(
    sentences: Sequence[str], out_path: str, epochs: int = 20, batch_size: int = 32,
    lr: float = 3e-3, device: str = "cpu",
) -> NeuralDiacritizer:
    """Train the neural tone/length tagger on tone-marked Hausa sentences."""
    model = NeuralDiacritizer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    samples = []
    for s in sentences:
        tones, lengths = NeuralDiacritizer.labels_from_marked(s)
        if len(tones) < 4:
            continue
        samples.append((s, tones, lengths))
    if not samples:
        raise ValueError("no usable annotated sentences")
    for ep in range(epochs):
        np.random.shuffle(samples)
        total = 0.0
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            max_len = max(len(c[1]) for c in chunk)
            ids = torch.zeros(len(chunk), max_len, dtype=torch.long, device=device)
            tgt_t = torch.zeros(len(chunk), max_len, dtype=torch.long, device=device)
            tgt_l = torch.zeros(len(chunk), max_len, dtype=torch.long, device=device)
            lens = torch.zeros(len(chunk), dtype=torch.long, device=device)
            for i, (s, t, l) in enumerate(chunk):
                emb_ids = model.encode_text(s)
                n = min(emb_ids.size(1), max_len)
                ids[i, :n] = emb_ids[0, :n]
                tgt_t[i, :n] = torch.tensor(t[:n])
                tgt_l[i, :n] = torch.tensor(l[:n])
                lens[i] = n
            tone_logits, len_logits = model.forward(ids, lens)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lens.unsqueeze(1)
            loss_t = F.cross_entropy(tone_logits[mask], tgt_t[mask])
            loss_l = F.cross_entropy(len_logits[mask], tgt_l[mask])
            loss = loss_t + loss_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach())
        print(f"[diacritizer] epoch {ep + 1}/{epochs} loss={total / max(1, len(samples) // batch_size):.4f}")
    torch.save({"model": model.state_dict()}, out_path)
    return model
