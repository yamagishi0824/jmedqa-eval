#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read configs/models.tsv (the evaluation registry).

  python src/registry.py plan [--only a,b] [--group g] [--sampling greedy|official] [--images]
      one run per line, tab-separated: name model profile mode tp sampling outdir kwargs (read by scripts/run_eval.sh)
  python src/registry.py list [same filters]
      status (done / todo) and output directory of each run

src/summarize.py uses load() to get labels and parameter counts.
"""
import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "configs" / "models.tsv"


def resolve_sampling(r: Dict[str, Any]) -> str:
    s = r.get("sampling") or "auto"
    if s != "auto":
        return s
    sys.path.insert(0, str(ROOT / "src"))
    from model_profiles import PROFILES
    prof = PROFILES[r["profile"]][r["mode"]]
    return "official" if prof.get("official_recommended", True) else "greedy"


def outdir_of(r: Dict[str, Any], images: bool, outputs_root: str = "outputs") -> str:
    kind = "vision_" if images else ""
    return f"{outputs_root}/{kind}{r['sampling']}/{r['name']}_{r['profile']}_{r['mode']}"


def load(path: Path = TSV) -> List[Dict[str, Any]]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"))
    for r in rows:
        for k, v in list(r.items()):
            if v == "-":
                r[k] = ""
        r["vision"] = r.get("vision") == "1"
        r["groups"] = [g for g in (r.get("groups") or "").split(",") if g]
        r["sampling"] = resolve_sampling(r)
        r["outdir"] = outdir_of(r, images=False)
        r["outdir_vision"] = outdir_of(r, images=True) if r["vision"] else ""
    seen = set()
    for r in rows:
        key = (r["name"], r["sampling"])
        if key in seen:
            raise ValueError(f"duplicate registry entry: {key}")
        seen.add(key)
    return rows


def select(rows, only: Optional[str] = None, group: Optional[str] = None,
           sampling: Optional[str] = None, images: bool = False) -> List[Dict[str, Any]]:
    names = set(only.split(",")) if only else None
    if names is not None:
        missing = names - {r["name"] for r in rows}
        if missing:
            raise SystemExit(f"[registry] not in configs/models.tsv: {sorted(missing)}")
    out = []
    for r in rows:
        if names is not None and r["name"] not in names:
            continue
        if group and group not in r["groups"]:
            continue
        if sampling and r["sampling"] != sampling:
            continue
        if images and not r["vision"]:
            continue
        out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["plan", "list"])
    p.add_argument("--only", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--sampling", default=None)
    p.add_argument("--images", action="store_true")
    a = p.parse_args()
    rows = select(load(), a.only or None, a.group or None, a.sampling or None, a.images)
    for r in rows:
        od = r["outdir_vision"] if a.images else r["outdir"]
        if a.cmd == "plan" and not r["model"]:
            continue  # results-only row (weights not public)
        if a.cmd == "plan":
            print("\t".join([r["name"], r["model"], r["profile"], r["mode"], r["tp"], r["sampling"], od, r["kwargs"] or "-"]))
        else:
            done = (ROOT / od / "jmedqa_stage1_full.jsonl").exists()
            print(f"{'done' if done else 'todo'}  {r['sampling']:8s} {r['name']:42s} {od}")


if __name__ == "__main__":
    main()
