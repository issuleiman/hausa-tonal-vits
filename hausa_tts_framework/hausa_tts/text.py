"""Hausa text front-end.

This module implements everything that happens *before* the acoustic model:

* orthographic normalisation of Boko (Latin) Hausa text,
* a phoneme inventory that treats tone and vowel length as first-class units,
* rule-based grapheme-to-phoneme (G2P) conversion that also extracts the
  tone (H / L / Falling / Raised-L) and the vowel length (short / long),
* post-lexical tone rules: low-tone raising, high-tone spreading and the
  pre-pausal neutralisation of the word-final length contrast,
* generation of *phonological F0 targets*, i.e. the pitch contour a Hausa
  speaker is expected to produce, including downdrift and declination.

Design notes / literature anchors
---------------------------------
* Tone is lexical in Hausa: each of the five vowels can carry high, low or
  falling tone (r12a, *Hausa (boko) orthography notes*).
* Vowel length is phonemic but under-represented in ordinary writing, and the
  word-final contrast is heavily reduced pre-pausally (Newman & Van Heuven 1981).
* Low-tone raising is best treated as *lexical/derived* and remains debated
  (Newman & Jaggar 1989; Schuh 1989) -- it is therefore a configurable rule.
* Hausa intonation = lexical tone + downdrift + sentence declination, with
  clause-final raising in questions (Inkelas, Leben & Cobler 1987;
  Lindau 1984; Rialland 2007; Li 2026).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. Inventory
# ---------------------------------------------------------------------------

#: Multi-character Boko graphemes must be matched before single characters.
CONSONANT_GRAPHEMES: List[str] = [
    # digraphs / labialised / palatalised series (Hausa has kw/gw, ky/gy, ƙw/ƙy)
    "ƙw", "gw", "kw", "ky", "gy", "ƙy", "sh", "ts",
    # simple consonants
    "b", "ɓ", "c", "d", "ɗ", "f", "g", "h", "j", "k", "ƙ", "l", "m", "n",
    "r", "s", "t", "w", "y", "z", "p", "ʼ",
]
VOWEL_GRAPHEMES: List[str] = ["a", "e", "i", "o", "u"]

#: Longest-match order used by the tokeniser.
_GRAPHEME_ORDER: List[str] = sorted(
    CONSONANT_GRAPHEMES + VOWEL_GRAPHEMES, key=len, reverse=True
)

#: Special model symbols.
SPECIALS: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>", "<sp>", "|"]
#: Silence / short-pause tokens produced by the front-end.
SIL_TALK = "<sp>"
SIL_PAUSE = "|"

#: Full phoneme inventory of the model (index == phoneme id).
PHONEMES: List[str] = SPECIALS + sorted(
    set(CONSONANT_GRAPHEMES) | set(VOWEL_GRAPHEMES) | {"ɑ"}
)
PHONEME2ID: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES)}
PAD, UNK, BOS, EOS = (
    PHONEME2ID["<pad>"],
    PHONEME2ID["<unk>"],
    PHONEME2ID["<bos>"],
    PHONEME2ID["<eos>"],
)

#: Tone symbols. ``H`` high, ``L`` low, ``F`` falling, ``R`` raised low
#: (low-tone raising), ``0`` unspecified.
TONES: List[str] = ["0", "H", "L", "F", "R"]
TONE2ID: Dict[str, int] = {t: i for i, t in enumerate(TONES)}

#: Vowel length symbols. ``-`` none (consonant), ``s`` short, ``l`` long.
LENGTHS: List[str] = ["-", "s", "l"]
LENGTH2ID: Dict[str, int] = {l: i for i, l in enumerate(LENGTHS)}

UTTERANCE_TYPES: List[str] = ["statement", "yesno", "wh"]
UTT2ID: Dict[str, int] = {u: i for i, u in enumerate(UTTERANCE_TYPES)}

#: ASCII conventions frequently used in Hausa corpora / keyboards.
_ASCII_TONE = {"@": "H", "!": "L", "^": "F"}
_ASCII_LENGTH = {":": "long", "~": "long"}

#: Combining marks (after NFD) -> tone / length they encode.
_COMBINING = {
    "\u0301": "H",      # acute accent
    "\u0300": "L",      # grave accent
    "\u0302": "F",      # circumflex (falling)
    "\u0304": "long",   # macron (length)
    "\u030c": "F",      # caron, sometimes used for falling
}


# ---------------------------------------------------------------------------
# 2. Token container
# ---------------------------------------------------------------------------


@dataclass
class PhonemeToken:
    """One segment of the phoneme sequence produced by the front-end."""

    phoneme: str
    tone: str = "0"
    length: str = "-"
    word_index: int = 0
    syllable_index: int = 0
    stress: float = 0.0
    #: True when the token is a pause/silence symbol.
    is_silence: bool = False
    #: True when the underlying length contrast was neutralised pre-pausally.
    length_neutralised: bool = False

    @property
    def is_vowel(self) -> bool:
        return self.phoneme in VOWEL_GRAPHEMES

    @property
    def pid(self) -> int:
        return PHONEME2ID.get(self.phoneme, UNK)

    @property
    def tid(self) -> int:
        return TONE2ID.get(self.tone, 0)

    @property
    def lid(self) -> int:
        return LENGTH2ID.get(self.length, 0)


# ---------------------------------------------------------------------------
# 3. Normalisation
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    "0": "sifili", "1": "ɗaya", "2": "biyu", "3": "uku", "4": "huɗu",
    "5": "biyar", "6": "shida", "7": "bakwai", "8": "takwas", "9": "tara",
    "10": "goma", "20": "ashirin", "30": "talatin", "40": "arba'in",
    "50": "hamsin", "100": "ɗari", "1000": "dubu",
}

_ABBREV = {
    "misali": "misali",
    "watau": "watau",
    "Dr.": "dakta",
    "Mr.": "mista",
    "Mrs.": "madam",
    "No.": "lamba",
    "km": "kilomita",
    "kg": "kilogram",
}


def normalise(text: str, expand_numbers: bool = True) -> str:
    """Normalise raw Hausa text into a canonical Boko string.

    * unicode NFKC, curly apostrophes -> ``ʼ``, remove unsupported marks,
    * collapse whitespace,
    * ``expand_numbers`` maps Arabic digits to Hausa number words (Hausa
      numerals are *not* read digit-by-digit).
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = (
        text.replace("\u2019", "ʼ")
        .replace("\u02bc", "ʼ")
        .replace("'", "ʼ")
        .replace("’", "ʼ")
    )
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    for k, v in _ABBREV.items():
        text = re.sub(rf"\b{re.escape(k)}", v, text, flags=re.IGNORECASE)
    if expand_numbers:
        def _sub(m: "re.Match[str]") -> str:
            tok = m.group(0)
            return _NUM_WORDS.get(tok, " ".join(_NUM_WORDS[c] for c in tok))

        text = re.sub(r"\d+", _sub, text)
    text = re.sub(r"[^\w\s\u0253\u0257\u0242\u0199\u02bc.,;:!?%\-\u0300-\u036f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _flatten_accents(text: str) -> List[Tuple[str, str, str]]:
    """Split a normalised string into ``(base_char, tone, length)`` triples.

    Handles both pre-composed characters (``á``) and decomposed sequences
    (``a`` + U+0301), plus the corpus conventions ``a:`` / ``a@`` / ``a!``.
    """
    decomposed = unicodedata.normalize("NFD", text)
    out: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(decomposed):
        ch = decomposed[i]
        tone, length = "0", "-"
        if ch in _COMBINING:  # stray combining mark, attach to previous char
            if out:
                b, t, l = out[-1]
                v = _COMBINING[ch]
                tone = v if v in TONE2ID else t
                length = "long" if v == "long" else l
                out[-1] = (b, tone, length)
            i += 1
            continue
        j = i + 1
        while j < len(decomposed) and decomposed[j] in _COMBINING:
            v = _COMBINING[decomposed[j]]
            if v in TONE2ID:
                tone = v
            else:
                length = v
            j += 1
        # trailing ASCII conventions:  a:  a@  a!
        if j < len(decomposed) and decomposed[j] in _ASCII_LENGTH:
            length = _ASCII_LENGTH[decomposed[j]]
            j += 1
        elif j < len(decomposed) and decomposed[j] in _ASCII_TONE:
            tone = _ASCII_TONE[decomposed[j]]
            j += 1
        out.append((ch, tone, length))
        i = j
    return out


# ---------------------------------------------------------------------------
# 4. G2P
# ---------------------------------------------------------------------------


def _tokenise_graphemes(word: str) -> List[Tuple[str, str, str]]:
    """Greedy longest-match tokenisation of a single word."""
    chars = _flatten_accents(word.lower())
    flat = "".join(c[0] for c in chars)
    meta = {i: (c[1], c[2]) for i, c in enumerate(chars)}
    toks: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(flat):
        matched = False
        for g in _GRAPHEME_ORDER:
            if flat.startswith(g, i):
                # a consonant grapheme never carries tone/length itself
                tone, length = ("0", "-")
                if g in VOWEL_GRAPHEMES:
                    tone, length = meta[i]
                toks.append((g, tone, length))
                i += len(g)
                matched = True
                break
        if not matched:
            toks.append((flat[i], "0", "-"))
            i += 1
    return toks


def g2p(
    text: str,
    *,
    low_tone_raising: bool = True,
    high_tone_spreading: bool = True,
    final_length_neutralisation: bool = True,
    pre_pausal: bool = True,
    syllable_marker: bool = False,
) -> List[PhonemeToken]:
    """Convert normalised Hausa text into a :class:`PhonemeToken` sequence.

    Punctuation becomes pause tokens: ``,``/``;``/``:`` -> short pause and
    ``.``/``!``/``?`` -> sentence-final pause, which matters because the
    word-final length contrast and the question intonation both depend on
    phrase-final position.
    """
    text = normalise(text)
    tokens: List[PhonemeToken] = []
    word_index = 0
    # split keeping punctuation as its own "word"
    for raw in re.findall(r"[^\s]+", text):
        if re.fullmatch(r"[.,;:!?%]+", raw):
            tokens.append(
                PhonemeToken(SIL_PAUSE if raw[0] in ".!?" else SIL_TALK, is_silence=True)
            )
            continue
        core = re.sub(r"^[^0-9a-zA-Z\u0253\u0257\u0199\u02bc]+|[^0-9a-zA-Z\u0253\u0257\u0199\u02bc]+$", "", raw)
        if not core:
            continue
        pieces = _tokenise_graphemes(core)
        syl = 0
        on_nucleus = False
        for g, tone, length in pieces:
            is_v = g in VOWEL_GRAPHEMES or g == "ɑ"
            if is_v and not on_nucleus:
                syl += 1
                on_nucleus = True
            elif is_v and on_nucleus:
                # diphthong / long vowel continuation: stays in same syllable
                length = "long" if length == "-" else length
            else:
                on_nucleus = False
            if is_v:
                length = "l" if length in ("long", "l") else ("s" if length in ("short", "s") else "-")
            tokens.append(
                PhonemeToken(
                    phoneme=g,
                    tone=tone if is_v else "0",
                    length=length if is_v else "-",
                    word_index=word_index,
                    syllable_index=syl,
                )
            )
        word_index += 1
        if syllable_marker:
            tokens.append(PhonemeToken(SIL_TALK, is_silence=True))

    tokens = apply_phonological_rules(
        tokens,
        low_tone_raising=low_tone_raising,
        high_tone_spreading=high_tone_spreading,
        final_length_neutralisation=final_length_neutralisation,
        pre_pausal=pre_pausal,
    )
    return tokens


# ---------------------------------------------------------------------------
# 5. Post-lexical phonology
# ---------------------------------------------------------------------------


def _word_spans(tokens: Sequence[PhonemeToken]) -> List[Tuple[int, int]]:
    """Group token indices into words using the ``word_index`` field.

    Splitting on ``word_index`` (rather than on pause tokens) is what makes the
    post-lexical rules correct for unpunctuated input, which is the common case
    for short Hausa prompts.
    """
    spans: List[Tuple[int, int]] = []
    start = None
    prev_word = None
    for i, t in enumerate(tokens):
        if t.is_silence:
            continue
        if start is None or t.word_index != prev_word:
            if start is not None:
                spans.append((start, i))
            start = i
        prev_word = t.word_index
    if start is not None:
        spans.append((start, len(tokens)))
    # trim trailing silence inside each span
    trimmed = []
    for a, b in spans:
        while b - 1 > a and tokens[b - 1].is_silence:
            b -= 1
        trimmed.append((a, b))
    return trimmed


def apply_phonological_rules(
    tokens: List[PhonemeToken],
    *,
    low_tone_raising: bool = True,
    high_tone_spreading: bool = True,
    final_length_neutralisation: bool = True,
    pre_pausal: bool = True,
) -> List[PhonemeToken]:
    """Apply post-lexical tone / length rules in place and return ``tokens``.

    * **Low-tone raising**: a low tone in word-final position is realised with
      a raised pitch target (marked ``R``). Its lexical vs. phonetic status is
      disputed (Newman & Jaggar 1989; Schuh 1989) -- hence the flag.
    * **High-tone spreading**: a high tone spreads one vowel to the right
      across a syllable boundary inside the same word.
    * **Final vowel-length neutralisation**: in pre-pausal position the
      short/long contrast of the word-final vowel is largely neutralised
      (Newman & Van Heuven 1981); the token keeps its *lexical* length but is
      flagged so the duration model can shorten it.
    """
    spans = _word_spans(tokens)
    if low_tone_raising:
        for a, b in spans:
            if tokens[a].tone == "L" and len(tokens) - a < 6:
                # raise a word-final low (also across a trailing vowel-only clitic)
                last_v = None
                for i in range(b - 1, a - 1, -1):
                    if tokens[i].is_vowel:
                        last_v = i
                        break
                if last_v is not None and tokens[last_v].tone == "L":
                    tokens[last_v].tone = "R"

    if high_tone_spreading:
        for a, b in spans:
            for i in range(a, b - 1):
                if tokens[i].is_vowel and tokens[i].tone == "H":
                    j = i + 1
                    while j < b and not tokens[j].is_vowel:
                        j += 1
                    if (
                        j < b
                        and tokens[j].tone == "L"
                        and tokens[j].syllable_index == tokens[i].syllable_index + 1
                    ):
                        tokens[j].tone = "H"

    if final_length_neutralisation and pre_pausal:
        for a, b in spans:
            # the word is pre-pausal if it is the last word or is followed by '|'
            rest = [t for t in tokens[b:] if not t.is_silence]
            followed_by_pause = (not rest) or any(t.phoneme == SIL_PAUSE for t in tokens[b:])
            if followed_by_pause:
                for i in range(b - 1, a - 1, -1):
                    if tokens[i].is_vowel:
                        if tokens[i].length == "l":
                            tokens[i].length_neutralised = True
                        break
    return tokens


# ---------------------------------------------------------------------------
# 6. Phonological F0 (pitch) targets
# ---------------------------------------------------------------------------


@dataclass
class ProsodyConfig:
    """Knobs of the rule-based Hausa pitch model (tuned on a development set)."""

    h_level: float = 0.00
    l_level: float = -0.34
    r_level: float = -0.16           # raised low
    f_start: float = -0.02           # falling tone start
    f_end: float = -0.48             # falling tone end
    declination: float = 0.16        # total downward drift across an utterance
    downdrift_step: float = 0.045    # each H after an L is lowered
    final_lengthening_pitch_drop: float = 0.06
    yesno_final_rise: float = 0.30
    yesno_global_raise: float = 0.05
    wh_global_raise: float = 0.14
    phrase_final_fall: float = 0.12


def f0_targets(
    tokens: Sequence[PhonemeToken],
    utt_type: str = "statement",
    cfg: Optional[ProsodyConfig] = None,
    utterance_index: int = 0,
) -> np.ndarray:
    """Return one log-F0 target per token (``0`` for consonants/silence).

    The targets are *relative* (speaker-independent) log-F0 values; the
    speaker-specific mean/std are applied later by the model so that the same
    contour can be produced by a male, female or child voice.
    """
    cfg = cfg or ProsodyConfig()
    n = len(tokens)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out

    # 1) lexical tone levels, with cumulative downdrift
    level = np.zeros(n, dtype=np.float32)
    drift = 0.0
    seen_low = False
    for i, t in enumerate(tokens):
        if not t.is_vowel:
            level[i] = cfg.h_level
            continue
        if t.tone == "L":
            level[i] = cfg.l_level - drift
            seen_low = True
        elif t.tone == "R":
            level[i] = cfg.r_level - drift
            seen_low = True
        elif t.tone == "F":
            level[i] = cfg.f_start - drift
        else:  # H or unspecified -> high (unmarked = high in Hausa convention)
            level[i] = cfg.h_level - drift
            if seen_low:
                drift += cfg.downdrift_step

    # 2) sentence declination: linear downward drift over the utterance
    pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
    level = level - cfg.declination * pos

    # 3) utterance-type effects
    if utt_type == "yesno":
        level = level + cfg.yesno_global_raise
        last_v = max((i for i, t in enumerate(tokens) if t.is_vowel), default=n - 1)
        level[last_v:] = level[last_v:] + cfg.yesno_final_rise
    elif utt_type == "wh":
        level = level + cfg.wh_global_raise
    # phrase-final fall on the very last vowel of a statement
    if utt_type == "statement":
        last_v = max((i for i, t in enumerate(tokens) if t.is_vowel), default=n - 1)
        level[last_v] = level[last_v] - cfg.phrase_final_fall

    # 4) falling tone is realised as a fall *within* the vowel: encode it as a
    #    two-point target by returning the mean and letting the vocoder hint
    #    include the slope (see `f0_hint_frames`).
    for i, t in enumerate(tokens):
        if t.is_vowel and t.tone == "F":
            level[i] = 0.5 * (cfg.f_start + cfg.f_end) - cfg.declination * pos[i]

    out = level.astype(np.float32)
    out[[t.is_silence for t in tokens]] = 0.0
    return out


def f0_hint_frames(
    tokens: Sequence[PhonemeToken],
    durations: Sequence[int],
    cfg: Optional[ProsodyConfig] = None,
    utt_type: str = "statement",
) -> np.ndarray:
    """Expand the per-phoneme pitch targets to a frame-level continuous hint.

    This is the pitch contour that the *vocoder* is conditioned on: it makes
    lexical tone, downdrift, declination and question intonation explicit
    rather than hoping the acoustic model infers them.
    """
    cfg = cfg or ProsodyConfig()
    levels = f0_targets(tokens, utt_type=utt_type, cfg=cfg)
    frames: List[float] = []
    for idx, (tok, dur) in enumerate(zip(tokens, durations)):
        dur = int(max(1, dur))
        if tok.is_vowel:
            lvl = float(levels[idx]) if idx < len(levels) else 0.0
            if tok.tone == "F":
                seg = np.linspace(lvl + 0.24, lvl - 0.24, dur, dtype=np.float32)
            elif tok.tone == "R":
                seg = np.linspace(cfg.r_level, cfg.r_level - 0.06, dur, dtype=np.float32)
            else:
                seg = np.full(dur, fill_value=lvl, dtype=np.float32)
            frames.extend(seg.tolist())
        else:
            # consonants interpolate towards the next vowel target (coarticulation)
            frames.extend([np.nan] * dur)
    arr = np.asarray(frames, dtype=np.float32)
    if arr.size == 0:
        return arr
    # linear interpolation across consonants / silence
    idx = np.arange(arr.size)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.zeros_like(arr)
    arr = np.interp(idx, idx[mask], arr[mask]).astype(np.float32)
    return arr


# ---------------------------------------------------------------------------
# 7. Convenience encoders
# ---------------------------------------------------------------------------


def encode(tokens: Sequence[PhonemeToken]) -> Dict[str, List[int]]:
    """Encode a token list into the integer tensors the model consumes."""
    return {
        "phoneme": [t.pid for t in tokens],
        "tone": [t.tid for t in tokens],
        "length": [t.lid for t in tokens],
        "syllable": [t.syllable_index for t in tokens],
        "word": [t.word_index for t in tokens],
    }


def tokens_to_text(tokens: Sequence[PhonemeToken], with_marks: bool = True) -> str:
    """Re-serialise tokens (useful for round-trip tests and debugging)."""
    accent = {"H": "\u0301", "L": "\u0300", "F": "\u0302", "R": "\u0300", "0": ""}
    out: List[str] = []
    for t in tokens:
        if t.is_silence:
            out.append(" " if t.phoneme == SIL_TALK else " . ")
        elif t.is_vowel and with_marks:
            base = t.phoneme
            s = base + accent[t.tone]
            if t.length == "l":
                s = base + "\u0304" + accent[t.tone]
            out.append(unicodedata.normalize("NFC", s))
        else:
            out.append(t.phoneme)
    return re.sub(r"\s+", " ", "".join(out)).strip()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    demo = "Yarinya tana̋ son abinci? Sunan ka wa ne."
    toks = g2p(demo)
    for t in toks:
        print(f"{t.phoneme:>4} tone={t.tone} len={t.length} syl={t.syllable_index}")
    print(tokens_to_text(toks))
    print(f0_targets(toks, utt_type="yesno").round(3))
