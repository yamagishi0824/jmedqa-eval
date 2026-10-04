#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based answer extraction.

The prompt asks for `Answer: X` on the last line (infer_jmedqa.ANSWER_LINE_*). parse_answer first reads that line
(strict) and otherwise tries common ways of stating a final answer (fallback). The rule that matched is returned as
`method`, so format compliance (method == "answer_line") can be reported as well.

  answer_line : the last "Answer: X" (English "Answer:" may appear mid-line; Japanese keywords such as
                解答 / 答え / 正解 / 最終解答 / 選択肢 followed by ":" / "：" only at the start of a line)
  boxed       : \\boxed{X}
  json        : "answer": "X"
  lead        : the first line of the final answer is only letters ("e" / "c, d") or exactly an option text
                ("b. 右腎部分切除術"), followed by an explanation
  last_line   : the last line is only letters / exactly an option text ("…です。\n\nc. 下痢")
  ja_phrase   : 「正解は X」「答えは X」「結論: X」 (strong), then 「よって X」「最も適切なものは X」 (weak); last occurrence of each
  bare        : the whole final answer is short (<= 100 chars) and starts with an option / number ("c", "c. ③", "**c**", "26"),
                or is only option texts
  none        : nothing found
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# ── normalization ───────────────────────────────────────────────────────

_MARKUP_RE = re.compile(r"[*_`#>]+")


def _norm(text: str) -> str:
    """Full-width alphanumerics to half-width (ａ→a, １→1) and strip markdown emphasis."""
    t = unicodedata.normalize("NFKC", text or "")
    return _MARKUP_RE.sub("", t)


def final_part(text: str) -> str:
    """Keep only the final answer if a reasoning part is present (gpt-oss "assistantfinal", </think>)."""
    s = text or ""
    m = list(re.finditer(r"assistant\s*final", s, flags=re.IGNORECASE))
    if m:
        s = s[m[-1].end():]
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    return s.strip()


# ── reading option letters / numbers ────────────────────────────────────

# separators between letters: , 、 ， / ・ と および 及び and & whitespace
_SEP = r"(?:\s*(?:,|、|，|/|・|&|and|と|および|及び)\s*|\s+)"
_OPEN = r"[\s\"'「『(（\[【:：✅☑✓→⇒-]*"


def _letters_at_start(seg: str, allowed: List[str]) -> List[str]:
    """Read a run of option letters at the start of seg: "c", "c, e", "(c)", "c. ③", "c（…）", "ce", ..."""
    allowed_set = "".join(sorted({a.lower() for a in allowed})) or "abcde"
    L = f"[{allowed_set}]"
    s = seg.strip()
    m = re.match(rf"{_OPEN}({L})(?![a-z])", s, flags=re.IGNORECASE)
    if not m:
        # letters written as one token, e.g. "ce"
        m2 = re.match(rf"{_OPEN}({L}{{2,5}})(?![a-z])", s, flags=re.IGNORECASE)
        if m2 and len(set(m2.group(1).lower())) == len(m2.group(1)):
            return sorted(set(m2.group(1).lower()))
        return []
    picked = [m.group(1).lower()]
    rest = s[m.end():]
    while True:
        m = re.match(rf"(?:の?「[^」]{{0,80}}」)?[\s\"'」』)）\]】.．:：]*{_SEP}{_OPEN}({L})(?![a-z])", rest, flags=re.IGNORECASE)
        if not m:
            break
        picked.append(m.group(1).lower())
        rest = rest[m.end():]
    return sorted(set(picked))


_NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _numbers_at_start(seg: str) -> List[str]:
    s = seg.strip().replace(",", "") if re.fullmatch(r"[\d,.\s]+", seg.strip()) else seg.strip()
    m = re.match(rf"{_OPEN}({_NUM_RE.pattern}(?:\s*(?:,|、|/)\s*{_NUM_RE.pattern})*)", s)
    return _NUM_RE.findall(m.group(1)) if m else []


def _canon_number(tok: str) -> str:
    s = tok.strip()
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return str(int(s))
        x = float(s)
        return str(int(x)) if x.is_integer() else ("%f" % x).rstrip("0").rstrip(".")
    except ValueError:
        return s


def _read(seg: str, mode: str, allowed: List[str]) -> str:
    if mode == "option":
        return ",".join(_letters_at_start(seg, allowed))
    return ",".join(_canon_number(x) for x in _numbers_at_start(seg))


# ── rules ───────────────────────────────────────────────────────────────

_ANSWER_LINE_RE = re.compile(
    # English "Answer:" may appear mid-line (not preceded by a letter); Japanese keywords only at the start of a line
    r"(?:(?<![A-Za-z])(?:final\s+answer|answer)|^[ \t]*(?:最終(?:的な)?(?:解答|回答|答え)|解答|回答|答え|正解|選択肢))"
    r"\s*[:：]\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_JSON_RE = re.compile(r"\"answer\"\s*:\s*\"([^\"]*)\"", flags=re.IGNORECASE)
# Strong phrases (explicit final answer), then weak ones (concluding conjunctions, 「適切な〜は」); last occurrence of each
_STRONG_PHRASE_RE = re.compile(
    r"(?<!他の)(?:正解|答え|解答|回答|正答|結論|正しい選択肢)(?:は|として(?:は)?)?\s*[:：]?\s*|"
    r"(?:the\s+)?(?:correct\s+|final\s+|best\s+)?(?:answer|choice|option)\s+(?:is|would\s+be)\s*:?\s*",
    flags=re.IGNORECASE,
)
_WEAK_PHRASE_RE = re.compile(
    r"(?:最も|もっとも)?(?:適切|正しい|妥当|考えられる|当てはまる|可能性が高い)な?(?:もの|選択肢|治療|検査|診断|所見|対応|処置|薬剤|薬|疾患)?(?:は|として(?:は)?)\s*[:：]?\s*|"
    r"(?:したがって|よって|従って|以上より|以上から|ゆえに|故に)[、,]?\s*|"
    r"(?:の|もの|所見|選択肢)は\s*[:：]?\s*",
    flags=re.IGNORECASE,
)


def _key(t: str) -> str:
    return re.sub(r"[\s。．.、,]+", "", _norm(t))


def _option_text_match(body: str, options: Optional[Dict[str, str]]) -> str:
    """Short outputs that give only option texts without letters ("黄色ブドウ球菌", "IgM 高値、抗ミトコンドリア抗体陽性")."""
    if not options:
        return ""
    b = _key(body)
    keys = {k.lower(): _key(str(v)) for k, v in options.items() if str(v).strip()}
    hit = [k for k, v in keys.items() if v and v == b]
    if hit:
        return hit[0]
    parts = [x for x in re.split(r"[、,，/・\n]|および|と", _norm(body)) if x.strip()]
    hits = []
    for part in parts:
        pk = _key(part)
        m = [k for k, v in keys.items() if v and v == pk]
        if not m:
            return ""
        hits.extend(m)
    return ",".join(sorted(set(hits)))


_BULLET_RE = re.compile(r"^[\s✅☑✓→⇒・\-]*")


def _collapse_echoes(body: str, options: Optional[Dict[str, str]]) -> str:
    """Collapse a letter followed by its own option text (「d. 肺炎球菌感染症」「d（肺炎球菌感染症）」) to the letter."""
    if not options:
        return body
    for k, v in sorted(options.items(), key=lambda kv: -len(str(kv[1]))):
        v = _norm(str(v)).strip()
        if len(v) < 2:
            continue
        body = re.sub(rf"(?<![A-Za-z])({re.escape(k)})\s*(?:[.．、:：]\s*{re.escape(v)}|[(（「]\s*{re.escape(v)}\s*[)）」])",
                      r"\1", body, flags=re.IGNORECASE)
    return body


def _last_line(body: str, mode: str, allowed: List[str], options: Optional[Dict[str, str]]) -> str:
    """The last line is only letters / exactly an option text ("…です。\n\nc. 下痢", "…\n\nd")."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < 2 or mode != "option":
        return ""
    return _lead("\n".join([lines[-1]]), mode, allowed, options)


def _echoed_options(lines: List[str], mode: str, options: Optional[Dict[str, str]], exact: bool = False) -> str:
    """Collect leading lines of the form "x. <option text>" (bullets / check marks ignored).
    exact=True requires the rest of the line to equal the option text (so headings like "a. Gaucher病: …" are ignored)."""
    if mode != "option" or not options:
        return ""
    picked: List[str] = []
    for ln in lines:
        t = _BULLET_RE.sub("", ln).strip()
        if not t:
            continue
        m = re.match(r"([a-z])\s*[.．、:：)）]\s*(.+)$", t, flags=re.IGNORECASE)
        k = m.group(1).lower() if m else ""
        opt = _norm(str(options.get(k, ""))).strip()
        if not m or not opt:
            break
        if exact and _key(m.group(2)) != _key(opt):
            break
        if not exact and not m.group(2).strip().startswith(opt[: max(4, min(len(opt), 12))]):
            break
        picked.append(k)
    return ",".join(sorted(set(picked)))


def _lead(body: str, mode: str, allowed: List[str], options: Optional[Dict[str, str]]) -> str:
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = _BULLET_RE.sub("", lines[0]).strip()
    if mode == "option":
        L = "".join(sorted({a.lower() for a in allowed})) or "abcde"
        if re.fullmatch(rf"\(?[{L}]\)?(?:\s*(?:,|、|，|/|・|と|and)\s*\(?[{L}]\)?)*[.。]?", first, flags=re.IGNORECASE):
            return ",".join(_letters_at_start(first, allowed))
        # "b – 母親も…" (letter + dash + explanation). ":" is excluded: it cannot be told apart from headings like "a: 〜は誤り"
        if re.match(rf"\(?[{L}]\)?\s+[–—―]\s+", first, flags=re.IGNORECASE):
            return ",".join(_letters_at_start(first, allowed))
        return _echoed_options(lines, mode, options, exact=True)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s*[^\s\d]{0,6}", first):  # "26" / "146 mg/dL" (unit up to 6 chars)
        return ",".join(_canon_number(x) for x in _numbers_at_start(first))
    return ""


def parse_answer(text: str, mode: str, allowed: List[str], options: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Return (prediction, method). prediction is normalized ("a" / "a,c" / "26"), or "" if nothing is found."""
    body = _norm(final_part(text))
    if not body:
        return "", "none"
    # "d. 肺炎球菌感染症 と e. …" -> "d と e"; used only for the Answer line and phrases
    # (applied globally, headings like "a. Gaucher病" would become bare letters and be picked up by `lead`)
    cbody = _collapse_echoes(body, options) if mode == "option" else body

    # 1) the last Answer: line
    for m in reversed(list(_ANSWER_LINE_RE.finditer(cbody))):
        pred = _read(m.group(1), mode, allowed)
        if pred:
            return pred, "answer_line"
    # 2) \boxed{}
    for m in reversed(list(_BOXED_RE.finditer(body))):
        pred = _read(m.group(1), mode, allowed)
        if pred:
            return pred, "boxed"
    # 3) "answer": "X"
    for m in reversed(list(_JSON_RE.finditer(body))):
        pred = _read(m.group(1), mode, allowed)
        if pred:
            return pred, "json"
    # 4) the first line is the answer (letters only / exactly an option text), followed by reasons
    pred = _lead(body, mode, allowed, options)
    if pred:
        return pred, "lead"
    pred = _last_line(body, mode, allowed, options)
    if pred:
        return pred, "last_line"
    # 5) Japanese / English final-answer phrases (strong -> weak, last occurrence of each)
    for rx in (_STRONG_PHRASE_RE, _WEAK_PHRASE_RE):
        for m in reversed(list(rx.finditer(cbody))):
            pred = _read(cbody[m.end():m.end() + 40], mode, allowed) or \
                _echoed_options(cbody[m.end():].splitlines()[:6], mode, options)
            if pred:
                return pred, "ja_phrase"
    # 6) the whole final answer is short and starts with an option / number ("c", "c. ③", "**c**", "26"), or is only option texts
    if len(body) <= 100:
        pred = _read(body.splitlines()[0], mode, allowed) or (
            _option_text_match(body, options) if mode == "option" else "")
        if pred:
            return pred, "bare"
    return "", "none"
