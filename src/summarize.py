#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headline tables for the runs in configs/models.tsv.

Accuracy is always reported on the 3,581 source questions:
  acc_original : original question text (image-referencing questions keep the text-only notice)
  acc_no_image : questions without image references use the original result; the 966 image-referencing
                 questions use their image-reference-removed (no_image) variant
  acc_image    : (vision runs) image-referencing questions answered with their images; the rest use the original result

Scoring: the rule-based answer from stage1 (src/answer_parser.py).

Outputs (in --out-dir):
  summary.csv                 one row per run
  summary_by_year.csv         accuracy per exam year (original, and with images for vision runs)
  vision_by_dependency.csv    image-referencing questions by image_dependency: original -> no_image -> with images
  summary.md                  Markdown tables for the README (all questions and exam year 2026; original text,
                              and with images for VLMs)
"""
import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from registry import ROOT, load

csv.field_size_limit(10**9)
Key = Tuple[str, str]  # (id, question_variant)


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _correct(outdir: Path) -> Optional[Tuple[Dict[Key, int], List[Dict[str, str]]]]:
    light = outdir / "jmedqa_stage1_light.csv"
    if not light.exists():
        return None
    rows = _read(light)
    corr = {(r["id"], r["question_variant"]): int(r["is_correct"]) for r in rows}
    return corr, rows


def _acc(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _f(x, missing: str = "") -> str:
    return missing if x == "" else f"{x:.3f}"


def markdown_tables(summary: List[Dict]) -> str:
    """README tables: main runs, reasoning-effort variants, and greedy comparison runs."""
    head = ("| Model | Category | Sampling | All (3,581) | 2026 (400) | With images: all | With images: 2026 |\n"
            "|---|---|---|---|---|---|---|\n")
    row = lambda s: (f"| {s['model']} | {s['subgroup'] or s['category']} | {s['sampling']} | {_f(s['acc_original'])} | "
                     f"{_f(s['acc_original_2026'])} | {_f(s['acc_image'], '–')} | {_f(s['acc_image_2026'], '–')} |\n")
    groups = lambda s: s["groups"].split(",")
    main = [s for s in summary if "effort" not in groups(s) and "t0cmp" not in groups(s)]
    effort = [s for s in summary if "effort" in groups(s)]
    t0 = [s for s in summary if "t0cmp" in groups(s)]
    md = "#### Main runs\n\n" + head + "".join(row(s) for s in main)
    if effort:
        md += "\n#### Reasoning-effort variants (default effort: see the main table)\n\n" + head + "".join(row(s) for s in effort)
    if t0:
        md += "\n#### Greedy runs of sampled models (for comparisons with greedy-only models)\n\n" + head + "".join(row(s) for s in t0)
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    summary, by_year, by_dep = [], [], []
    for r in load():
        res = _correct(ROOT / r["outdir"])
        if res is None:
            continue
        corr, rows = res
        orig = {k[0]: v for k, v in corr.items() if k[1] == "original"}
        noimg = dict(orig)
        noimg.update({k[0]: v for k, v in corr.items() if k[1] == "no_image"})
        meta = {x["id"]: x for x in rows if x["question_variant"] == "original"}
        img = None
        if r["vision"]:
            vres = _correct(ROOT / r["outdir_vision"])
            if vres is not None:
                img = dict(orig)
                img.update({k[0]: v for k, v in vres[0].items() if k[1] == "original_image"})
        o_rows = list(meta.values())
        tok = lambda col: st.median(int(x[col]) for x in o_rows) if o_rows and o_rows[0].get(col, "") != "" else ""
        summary.append({
            "model": r["name"], "category": r["category"], "subgroup": r["subgroup"], "family": r["family"],
            "sampling": r["sampling"], "n_questions": len(orig),
            "acc_original": round(_acc(orig.values()), 4), "acc_no_image": round(_acc(noimg.values()), 4),
            "acc_image": round(_acc(img.values()), 4) if img else "",
            "answer_line_rate": round(_acc(int(x.get("rule_method") == "answer_line") for x in o_rows), 4),
            "extraction_success": round(_acc(int(x.get("prediction", "") != "") for x in o_rows), 4),
            "truncated": sum(x.get("finish_reason") == "length" for x in o_rows),
            "reasoning_tokens_median": tok("n_reasoning_tokens"), "content_tokens_median": tok("n_content_tokens"),
            "total_params_B": r["total_B"], "active_params_B": r["active_B"],
            "acc_original_2026": round(_acc(v for q, v in orig.items() if meta[q]["year"] == "2026"), 4),
            "acc_image_2026": round(_acc(v for q, v in img.items() if meta[q]["year"] == "2026"), 4) if img else "",
            "groups": ",".join(r["groups"]),
        })
        years = defaultdict(list)
        for qid, v in orig.items():
            years[meta[qid]["year"]].append(qid)
        for y in sorted(years):
            ids = years[y]
            by_year.append({"model": r["name"], "sampling": r["sampling"], "year": y, "n": len(ids),
                            "acc_original": round(_acc(orig[i] for i in ids), 4),
                            "acc_image": round(_acc(img[i] for i in ids), 4) if img else ""})
        if img:
            vids = [k[0] for k in vres[0] if k[1] == "original_image"]
            groups = defaultdict(list)
            for i in vids:
                groups[meta[i]["image_dependency"]].append(i)
            for dep, ids in sorted(groups.items()):
                by_dep.append({"model": r["name"], "image_dependency": dep, "n": len(ids),
                               "acc_original": round(_acc(orig[i] for i in ids), 4),
                               "acc_no_image": round(_acc(noimg[i] for i in ids), 4),
                               "acc_image": round(_acc(img[i] for i in ids), 4)})

    summary.sort(key=lambda x: -x["acc_original"])
    for name, rows_ in (("summary.csv", summary), ("summary_by_year.csv", by_year), ("vision_by_dependency.csv", by_dep)):
        if rows_:
            with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
                w.writeheader()
                w.writerows(rows_)
    (out / "summary.md").write_text(markdown_tables(summary), encoding="utf-8")
    print(f"| model | sampling | original | no_image | image |\n|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['model']} | {s['sampling']} | {s['acc_original']:.3f} | {s['acc_no_image']:.3f} | "
              f"{s['acc_image'] if s['acc_image'] == '' else format(s['acc_image'], '.3f')} |")
    print(f"\n[OK] {len(summary)} runs -> {out}")


if __name__ == "__main__":
    main()
