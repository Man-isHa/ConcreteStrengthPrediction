"""
Step 2 - Parse the packed cells and explode to a tidy long table.

The Compressive Strength cells pack a grid of (replacement %, curing age) -> MPa
in ~12 different textual layouts. We extract every (replacement_pct, age_days,
strength_mpa) triple we can confidently identify and write:

  out/train_long.csv       one row per (paper_id, replacement_pct, age_days)
  out/parse_failures.csv   cells (or segments) we could not parse, for review

Design: precision over recall. A triple is emitted only when BOTH a replacement
level and a curing age can be resolved (from an inline qualifier, a segment
header, or a persistent line context). Bare number lists with no age/% mapping
are logged as failures rather than guessed.

Run: python pipeline/02_parse_explode.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WIDE = OUT / "papers_wide.csv"

CS_COL = "Compressive Strength::Extracted Value"
LEVEL_COL = "Cement Replacement Level(s)::Extracted Value"

# --- regexes ---------------------------------------------------------------
# Unsigned: replacement %, curing age and strength are never negative. A leading
# '-' in the text is always a separator (e.g. "RHA8-10%", "Aa-30"), not a sign.
NUM = r"\d+(?:\.\d+)?"
AGE_RE = re.compile(rf"({NUM})\s*[-\s]*(?:days?|d)\b", re.I)
PCT_EXPLICIT_RE = re.compile(rf"({NUM})\s*%")
# mix-label encodings of replacement %
MIX_RHA_RE = re.compile(rf"\bR?HA[\s\-]?({NUM})\b", re.I)   # RHA10, RHA-2.5, HA5
MIX_R_RE = re.compile(rf"\bR[\s\-]?(\d+(?:\.\d+)?)(?:W\d+)?\b")  # R0, R10, R10W35
MIX_CR_RE = re.compile(rf"\bCR[\s\-]?({NUM})\b", re.I)      # CR5, CR10
MIX_NUM_RHA_RE = re.compile(rf"\b({NUM})\s*RHA\b", re.I)    # 0RHA, 5RHA
MIX_MDASH_RE = re.compile(rf"\bM[\-_]({NUM})\b", re.I)      # M-15, M-25
VALUE_RE = re.compile(NUM)
CONTROL_RE = re.compile(r"\b(control|ctrl|ctl|cm|nc|opc|reference|ref|c-?m)\b", re.I)

# tokens that mean "this is a value, not an age/pct"
RANGE_SEP = re.compile(r"\bto\b|–|—|\x96|~|\s-\s")

MAX_REASONABLE_MPA = 250.0
MIN_REASONABLE_MPA = 0.5


def find_age(s):
    m = AGE_RE.search(s)
    return float(m.group(1)) if m else None


def find_pct(s, allowed=None):
    """Resolve a replacement percentage from a string, or None. Precision-first.

    If ``allowed`` (the paper's declared levels) is given, a final fallback picks a
    bare number that matches a declared level - this safely recovers mix labels
    like "Aa-20" or "A35-10" without guessing unmappable IDs (M1, RHA3).
    """
    m = PCT_EXPLICIT_RE.search(s)
    if m:
        return float(m.group(1))
    # control keywords -> 0% replacement
    if CONTROL_RE.search(s):
        # guard: 'C-RHA7' style labels contain RHA -> handled below, not control
        if not re.search(r"RHA\d", s, re.I):
            return 0.0
    for rx in (MIX_NUM_RHA_RE, MIX_RHA_RE, MIX_CR_RE, MIX_MDASH_RE, MIX_R_RE):
        m = rx.search(s)
        if m:
            return float(m.group(1))
    # whitelist-guided fallback: any bare number equal to a declared level
    if allowed:
        no_age = AGE_RE.sub(" ", s)
        for tok in VALUE_RE.findall(no_age):
            v = float(tok)
            if v != 0.0 and any(abs(v - a) < 0.01 for a in allowed):
                return v
    return None


def parse_level_list(cell):
    """Parse the 'Cement Replacement Level(s)' cell into a set of declared %.

    Returns None if the cell is empty/unparseable (=> no whitelist for that paper).
    """
    if not isinstance(cell, str) or not cell.strip():
        return None
    nums = re.findall(NUM, cell.replace("%", " "))
    vals = {float(n) for n in nums if 0.0 <= float(n) <= 100.0}
    return vals or None


def split_items(body):
    """Split a segment body into individual value items."""
    return [it for it in re.split(r"[,|]", body) if it.strip()]


def is_range(item):
    return bool(RANGE_SEP.search(item))


def parse_cs_cell(text, allowed_levels=None):
    """Return (triples, unparsed_segments).

    triples: list of dict(replacement_pct, age_days, strength_mpa, is_range).
    If ``allowed_levels`` is given, a resolved replacement % must be in it (0 for
    control is always allowed); mislabelled sample IDs (e.g. RHA1/M2) are rejected.
    """
    triples, unparsed = [], []
    if not isinstance(text, str):
        return triples, unparsed

    # normalize bullets / unicode
    text = text.replace("\x95", "\n").replace("•", "\n").replace("\r", "\n")

    segments = [s for s in re.split(r"[;\n]+", text) if s.strip()]
    ctx_age = None      # persists across lines (e.g. a lone "28-day:" header line)
    ctx_pct = None
    for seg in segments:
        seg = seg.strip(" \t-•*")
        if not seg:
            continue

        # split header (before first ':') from body
        head, sep, body = seg.partition(":")
        seg_age = seg_pct = None
        if sep:
            seg_age = find_age(head)
            seg_pct = find_pct(head, allowed=allowed_levels)
        else:
            body = seg  # no header; whole segment is body

        # A lone header line ("28-day:" with empty/qualifier-only body) sets context.
        if sep and not VALUE_RE.search(body):
            if seg_age is not None:
                ctx_age = seg_age
            if seg_pct is not None:
                ctx_pct = seg_pct
            continue
        # A header that carries age/pct updates persistent context too.
        if seg_age is not None:
            ctx_age = seg_age
        if seg_pct is not None:
            ctx_pct = seg_pct

        items = split_items(body)
        for item in items:
            it = item.strip()
            if not it:
                continue
            item_age = find_age(it)
            # value = first number that is NOT the age/pct qualifier
            # strip age/pct tokens then read the leading value
            cleaned = AGE_RE.sub(" ", it)
            cleaned = PCT_EXPLICIT_RE.sub(" ", cleaned)
            vals = VALUE_RE.findall(cleaned)
            item_pct = find_pct(it, allowed=allowed_levels)

            age = item_age if item_age is not None else (seg_age if seg_age is not None else ctx_age)
            pct = item_pct if item_pct is not None else (seg_pct if seg_pct is not None else ctx_pct)

            if not vals or age is None or pct is None:
                unparsed.append(it)
                continue

            rng = is_range(it)
            if rng and len(vals) >= 2:
                nums = [float(v) for v in vals[:2]]
                value = sum(nums) / 2.0
            else:
                value = float(vals[0])

            if not (MIN_REASONABLE_MPA <= value <= MAX_REASONABLE_MPA):
                unparsed.append(it)
                continue
            if not (0.0 <= pct <= 100.0) or not (0.0 < age <= 400.0):
                unparsed.append(it)
                continue
            # whitelist cross-check against the paper's declared replacement levels
            if allowed_levels is not None and pct != 0.0:
                if not any(abs(pct - a) < 0.01 for a in allowed_levels):
                    unparsed.append(it + "  [pct not in declared levels]")
                    continue
            triples.append(dict(replacement_pct=pct, age_days=age,
                                strength_mpa=round(value, 3), is_range=rng))
    return triples, unparsed


def main():
    w = pd.read_csv(WIDE)
    rows, failures = [], []
    papers_with_cs = 0
    papers_yielding = 0

    for _, r in w.iterrows():
        cell = r[CS_COL]
        if not isinstance(cell, str) or not cell.strip():
            continue
        papers_with_cs += 1
        allowed = parse_level_list(r[LEVEL_COL]) if LEVEL_COL in w.columns else None
        triples, unparsed = parse_cs_cell(cell, allowed_levels=allowed)
        if triples:
            papers_yielding += 1
            for t in triples:
                rows.append({"paper_id": r["paper_id"], **t})
        for u in unparsed:
            failures.append({"paper_id": r["paper_id"], "segment": u,
                             "cell_preview": cell[:120]})

    long = pd.DataFrame(rows).drop_duplicates()
    long.to_csv(OUT / "train_long.csv", index=False, encoding="utf-8")
    pd.DataFrame(failures).to_csv(OUT / "parse_failures.csv", index=False, encoding="utf-8")

    print(f"papers with CS cell : {papers_with_cs}")
    print(f"papers that yielded : {papers_yielding}")
    print(f"CS data points      : {len(long)}")
    print(f"unparsed segments   : {len(failures)}")
    if len(long):
        print("\nage distribution (days):")
        print(long["age_days"].value_counts().sort_index().to_string())
        print("\nreplacement %% distribution (top):")
        print(long["replacement_pct"].value_counts().sort_index().head(20).to_string())
        print(f"\nstrength range: {long.strength_mpa.min()}-{long.strength_mpa.max()} MPa")
        print("\nsample rows:")
        print(long.head(12).to_string())


if __name__ == "__main__":
    main()
