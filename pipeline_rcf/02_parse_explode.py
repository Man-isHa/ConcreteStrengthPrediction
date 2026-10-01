"""
Step 2 - parse the packed strength cells into a tidy long table.

Two sources feed one schema:
  A. Sheet1 "Compressive Strength" - free text packing a
     (replacement level x curing age [x w/b]) grid in ~16 layouts.
  B. Experimental Matrix "Reported Strength" - one row per mix, with the
     replacement level supplied by its own column.

Design principle, inherited from the RHA pipeline: precision over recall. A row
is emitted only when the replacement level resolves; the curing age is allowed to
be unknown and is flagged, because most matrix cells omit it and assuming 28 d
would be fabrication. Study 6 in 06_ablation.py tests that assumption empirically.

Run: python pipeline_rcf/02_parse_explode.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NUM, OUT, is_null  # noqa: E402

CS_COL = "Compressive Strength::Extracted Value"
UNIT_COL = "Compressive Strength::Unit"
LEVEL_COL = "Cement Replacement Level(s)::Extracted Value"
MATERIAL_COL = "Material Type::Extracted Value"

# ------------------------------------------------------------------ regexes
NUM_RE = re.compile(NUM)
AGE_RE = re.compile(rf"({NUM})\s*[-\s]?\s*(?:d\b|days?\b|day\b)", re.I)
PCT_RE = re.compile(rf"({NUM})\s*(?:%|wt\.?-?%)")
WB_RE = re.compile(rf"w\s*/\s*[bc]\s*[=:]?\s*({NUM})", re.I)
# a bare 0.20-0.79 inside a mix label is a water/binder ratio in this corpus
LABEL_WB_RE = re.compile(r"(?<![\d.])(0\.[2-7]\d?)(?![\d])")
SD_RE = re.compile(rf"(?:±|\+/-|\+-)\s*{NUM}")
CONTROL_RE = re.compile(
    r"\b(ref|reference|control|opc|nac|baseline|plain|ordinary|blank|"
    r"pc|cem\s*i|c0|s0|m0|p0|r0|n0)\b", re.I)
GRAPH_RE = re.compile(r"graph[-\s]?derived|approx|~", re.I)
FLEX_RE = re.compile(r"flexur|split|tensile|modulus|elastic|shrink|slump", re.I)
COMP_RE = re.compile(r"compress", re.I)

MPA_PER_PSI = 0.00689476
MPA_PER_KGCM2 = 0.0980665

MIN_MPA, MAX_MPA = 0.5, 250.0
MAX_AGE = 400.0

# Material families that are not fines/powder of crushed concrete or C&D waste.
OFFSCOPE_RE = re.compile(r"\b(brick|ceramic|glass|tile|aerated|aac|"
                         r"fly\s*ash|slag\s*only|rice\s*husk)\b", re.I)
INSCOPE_RE = re.compile(r"concrete|cement|cdw|c&d|demolition|mortar|paste", re.I)


def unit_factor(unit):
    """MPa per reported unit. Ambiguous compound units are treated as MPa."""
    if is_null(unit):
        return 1.0
    u = str(unit).strip().lower()
    if u == "psi":
        return MPA_PER_PSI
    if u.startswith("kg/cm"):
        return MPA_PER_KGCM2
    return 1.0


def parse_level_list(cell):
    """The paper's declared replacement levels, used as a whitelist.

    Handles '10%, 20%, 30%', '10; 20; 30', '0, 5, 7.5, ...',
    'Mortars: 0%, 15%, 25%; Concretes: 0%, 15%, 25%'.
    """
    allowed = {0.0}
    if is_null(cell):
        return allowed
    for v in NUM_RE.findall(str(cell)):
        f = float(v)
        if 0.0 <= f <= 100.0:
            allowed.add(f)
    return allowed


def strip_sd(text):
    return SD_RE.sub(" ", text)


def find_age(text):
    m = AGE_RE.search(text)
    if not m:
        return None
    a = float(m.group(1))
    return a if 0 < a <= MAX_AGE else None


def find_pct(text, allowed):
    """Resolve a replacement level from a label or qualifier.

    Order: explicit percent -> control keyword -> a number embedded in a mix ID
    that matches one of the paper's declared levels.
    """
    m = PCT_RE.search(text)
    if m:
        f = float(m.group(1))
        return f if 0.0 <= f <= 100.0 else None
    if CONTROL_RE.search(text):
        return 0.0
    # mix labels such as RCCF 25, C30R, MA20, S10
    for v in NUM_RE.findall(text):
        f = float(v)
        if f in allowed and 0.0 <= f <= 100.0:
            return f
    return None


def emit(rows, val, pct, age, wb, graph, source, factor, paper_id=None,
         mix_id=None, mix_activation=None):
    """Append one clamped row. Returns True when the row survived the clamps."""
    v = val * factor
    if not (MIN_MPA <= v <= MAX_MPA):
        return False
    if pct is None or not (0.0 <= pct <= 100.0):
        return False
    if age is not None and not (0 < age <= MAX_AGE):
        age = None
    rows.append(dict(paper_id=paper_id, mix_id=mix_id, replacement_pct=pct,
                     age_days=age, strength_mpa=round(v, 3), wb_block=wb,
                     graph_derived=graph, age_known=age is not None,
                     mix_activation=mix_activation, source=source))
    return True


def split_item(item):
    """Split one item into (label, value_zone).

    Two shapes occur, and the strength lives in a different place in each:
      "R20: 48.1"        -> label before the colon, value after
      "33.2 (MA20)"      -> value outside the parentheses, label inside
    """
    if ":" in item:
        label, _, value = item.rpartition(":")
        return label, re.sub(r"\([^)]*\)", " ", value)
    label = " ".join(re.findall(r"\(([^)]*)\)", item))
    return label, re.sub(r"\([^)]*\)", " ", item)


def parse_cs_cell(text, allowed, factor):
    """Sheet1 layout parser. Returns (rows, unparsed_segments)."""
    rows, unparsed = [], []
    ctx_age = ctx_pct = ctx_wb = None

    for raw_line in str(text).replace("\r", "").split("\n"):
        line = raw_line.replace("•", " ").strip(" \t-")
        if not line:
            continue

        wb = WB_RE.search(line)
        if wb:
            ctx_wb = float(wb.group(1))

        head, sep, tail = line.partition(":")
        if sep and not NUM_RE.search(tail):
            # pure header line: "Ref (0%):", "28 Days:", "w/b 0.4:"
            a, p = find_age(head), find_pct(head, allowed)
            if a is not None:
                ctx_age = a
            if p is not None:
                ctx_pct = p
            if a is None and p is None and not WB_RE.search(head):
                ctx_pct = None      # an unrecognised header must not leak context
            continue

        line_age, line_pct = ctx_age, ctx_pct
        if sep and tail.strip():
            # "3-Day: 33.2 (MA20), ..." - the prefix scopes the rest of the line
            a, p = find_age(head), find_pct(head, allowed)
            if a is not None:
                line_age = a
            if p is not None:
                line_pct = p
            body = tail
        else:
            body = line

        got_any = False
        for item in re.split(r"[;,]", body):
            item = item.strip()
            if not item or not NUM_RE.search(item):
                continue
            graph = bool(GRAPH_RE.search(item)) or bool(GRAPH_RE.search(line))
            label, value_zone = split_item(strip_sd(item))

            # Age and level may be stated on either side; the strength is only
            # ever read from the value zone. Reading it from the whole item lets
            # a mix label such as "R20:" or "N-0.50-0:" masquerade as the value.
            for text in (label, value_zone):
                a = find_age(text)
                if a is not None:
                    line_age = a
                p = find_pct(text, allowed)
                if p is not None:
                    line_pct = p
                    break

            # mix labels of the form "N-0.40-0" / "P-0.45-16" carry the w/b ratio
            wb = ctx_wb
            mwb = LABEL_WB_RE.search(label)
            if mwb:
                wb = float(mwb.group(1))

            consumed = set()
            for m in AGE_RE.finditer(value_zone):
                consumed.add(m.group(1))
            for m in PCT_RE.finditer(value_zone):
                consumed.add(m.group(1))
            vals = [v for v in NUM_RE.findall(value_zone) if v not in consumed]
            if not vals:
                continue
            val = float(vals[0])
            if emit(rows, val, line_pct, line_age, wb, graph, "sheet1", factor,
                    mix_id=label.strip() or None):
                got_any = True

        if not got_any:
            unparsed.append(line[:200])

    return rows, unparsed


# ------------------------------------------------------------------ source B
MPA_ITEM_RE = re.compile(rf"({NUM})\s*MPa", re.I)


def parse_matrix_level(cell):
    """Replacement level from the matrix's own column.

    Returns (pct, reason) - reason is non-empty when the row is out of scope.
    """
    if is_null(cell):
        return None, "no level"
    s = str(cell)
    if re.search(r"\bsand\b|fine\s+natural\s+aggregate|coarse\s+aggregate|"
                 r"\bfra\b|\bfrca\b|aggregate\s+replacement", s, re.I):
        # sand/aggregate replacement is a different intervention from cement
        # replacement; mixing the two would confound the target
        if not re.search(r"cement|binder|paste|powder", s, re.I):
            return None, "aggregate replacement"
    m = PCT_RE.search(s)
    if m:
        return float(m.group(1)), ""
    nums = NUM_RE.findall(s)
    if not nums:
        return None, "no number"
    f = float(nums[0])
    if f <= 1.0 and "." in nums[0]:
        f *= 100.0            # '0.3' means 30%, not 0.3%
    return (f, "") if 0.0 <= f <= 100.0 else (None, "out of range")


def parse_matrix_strength(text):
    """Compressive strengths from a matrix cell, as (value_mpa, age_or_None).

    Requires an explicit 'MPa' token, which conveniently excludes the
    relative-only cells ('113.1% of untreated RCF mortar', '3-day: +18%').
    """
    out = []
    s = str(text).replace("\n", "; ")
    for seg in re.split(r"[;/]", s):
        if not MPA_ITEM_RE.search(seg):
            continue
        if FLEX_RE.search(seg) and not COMP_RE.search(seg):
            continue
        clean = strip_sd(seg)
        age = find_age(clean)
        for m in MPA_ITEM_RE.finditer(clean):
            out.append((float(m.group(1)), age))
    return out


AGG_RE = re.compile(r"aggregate|\bsand\b", re.I)
CEMENT_REPL_RE = re.compile(r"replacement\s+of\s+cement|cement\s+replacement|"
                            r"binder\s+replacement|replacing\s+cement", re.I)


def material_in_scope(cell):
    if is_null(cell):
        return True
    s = str(cell)
    if OFFSCOPE_RE.search(s) and not INSCOPE_RE.search(s):
        return False
    return True


def levels_in_scope(cell):
    """Reject papers whose declared levels are an aggregate/sand substitution.

    Those studies replace sand rather than cement, so their 'replacement level'
    means something different from the rest of the corpus. Pooling them would
    reintroduce exactly the confound that makes framing A unlearnable.
    """
    if is_null(cell):
        return True
    s = str(cell)
    if AGG_RE.search(s) and not CEMENT_REPL_RE.search(s):
        return False
    return True


def main():
    wide = pd.read_csv(OUT / "papers_wide.csv")
    mx = pd.read_csv(OUT / "mix_matrix.csv")

    scope = {int(r.paper_id): (material_in_scope(r.get(MATERIAL_COL))
                               and levels_in_scope(r.get(LEVEL_COL)))
             for _, r in wide.iterrows()}
    excluded = sorted(p for p, ok in scope.items() if not ok)
    pd.DataFrame([
        dict(paper_id=p,
             title=str(wide.loc[wide.paper_id == p, "Paper Title"].iloc[0])[:120],
             material=wide.loc[wide.paper_id == p, MATERIAL_COL].iloc[0],
             levels=wide.loc[wide.paper_id == p, LEVEL_COL].iloc[0])
        for p in excluded
    ]).to_csv(OUT / "excluded_papers.csv", index=False, encoding="utf-8")

    allowed_by_paper = {int(r.paper_id): parse_level_list(r.get(LEVEL_COL))
                        for _, r in wide.iterrows()}

    rows, failures = [], []
    n_cells_a = 0
    for _, r in wide.iterrows():
        pid = int(r.paper_id)
        cell = r.get(CS_COL)
        if is_null(cell) or not scope[pid]:
            continue
        n_cells_a += 1
        got, unparsed = parse_cs_cell(cell, allowed_by_paper[pid],
                                      unit_factor(r.get(UNIT_COL)))
        for g in got:
            g["paper_id"] = pid
            rows.append(g)
        for u in unparsed:
            failures.append(dict(paper_id=pid, source="sheet1", segment=u))

    n_cells_b = 0
    for _, r in mx.iterrows():
        pid = int(r.paper_id)
        if not scope.get(pid, True):
            continue
        cell = r.get("Reported Strength")
        if is_null(cell):
            continue
        n_cells_b += 1
        pct, reason = parse_matrix_level(r.get("Replacement Level"))
        vals = parse_matrix_strength(cell)
        if pct is None or not vals:
            failures.append(dict(paper_id=pid, source="matrix",
                                 segment=f"[{reason or 'no MPa'}] "
                                         f"{str(cell)[:160]}"))
            continue
        graph = bool(GRAPH_RE.search(str(cell)))
        mix_id = None if is_null(r.get("Mix ID")) else str(r.get("Mix ID")).strip()
        act = (None if is_null(r.get("Activation Method"))
               else str(r.get("Activation Method")).strip())
        for val, age in vals:
            emit(rows, val, pct, age, None, graph, "matrix", 1.0, paper_id=pid,
                 mix_id=mix_id, mix_activation=act)

    long = pd.DataFrame(rows)
    long = long[["paper_id", "mix_id", "replacement_pct", "age_days",
                 "strength_mpa", "wb_block", "graph_derived", "age_known",
                 "mix_activation", "source"]]

    # The two sources overlap: the same measurement is often in both. Dedupe on
    # the *measurement*, not on (paper, level, age) - several distinct mixes can
    # share a replacement level at one age and differ only by activation, and
    # collapsing those would throw away most of the matrix.
    n_pre = len(long)
    long["_pri"] = (long.source == "matrix").astype(int)   # Sheet1 wins ties
    long = (long.sort_values(["paper_id", "replacement_pct", "age_days",
                              "strength_mpa", "_pri"])
                .drop_duplicates(["paper_id", "replacement_pct", "age_days",
                                  "strength_mpa"], keep="first")
                .drop(columns="_pri")
                .reset_index(drop=True))
    print(f"dedup                     : {n_pre} -> {len(long)} rows")

    long.to_csv(OUT / "train_long.csv", index=False, encoding="utf-8")
    pd.DataFrame(failures).to_csv(OUT / "parse_failures.csv", index=False,
                                  encoding="utf-8")

    aged = long[long.age_known]
    yield_rows = [
        dict(stage="papers in workbook", n=len(wide)),
        dict(stage="papers in scope", n=len(wide) - len(excluded)),
        dict(stage="papers with a strength cell", n=n_cells_a),
        dict(stage="papers yielding >=1 point", n=long.paper_id.nunique()),
        dict(stage="papers yielding age-resolved points", n=aged.paper_id.nunique()),
    ]
    pd.DataFrame(yield_rows).to_csv(OUT / "parse_yield.csv", index=False,
                                    encoding="utf-8")

    print(f"off-scope papers excluded : {len(excluded)} -> {excluded}")
    print(f"source A cells parsed     : {n_cells_a}")
    print(f"source B cells parsed     : {n_cells_b}")
    print(f"total points              : {len(long)} "
          f"over {long.paper_id.nunique()} papers")
    print(f"  age-resolved            : {len(aged)} "
          f"over {aged.paper_id.nunique()} papers")
    print(f"  by source               : {long.source.value_counts().to_dict()}")
    print(f"  at 28 d                 : {(long.age_days == 28).sum()}")
    print(f"  strength range          : {long.strength_mpa.min():.1f} - "
          f"{long.strength_mpa.max():.1f} MPa")
    print(f"parse failures logged     : {len(failures)}")

    ctrl = aged[aged.replacement_pct == 0].paper_id.nunique()
    print(f"papers with a 0% control (age-resolved): {ctrl}")


if __name__ == "__main__":
    main()
