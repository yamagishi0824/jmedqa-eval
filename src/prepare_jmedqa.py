#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_options(options: Any) -> Dict[str, str]:
    if not isinstance(options, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in options.items():
        kk = normalize_text(k).lower()
        if kk in {"a", "b", "c", "d", "e"}:
            out[kk] = normalize_text(v)

    for kk in ["a", "b", "c", "d", "e"]:
        out.setdefault(kk, "")

    return out


def normalize_answer(answer: Any) -> List[str]:
    if not isinstance(answer, list):
        return []

    out: List[str] = []
    for x in answer:
        s = normalize_text(x)
        if not s:
            continue
        out.append(s)
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error: {path}:{line_no}: {e}") from e
    return rows


def clean_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cleaned: List[Dict[str, Any]] = []

    stats = {
        "input_rows": len(rows),
        "dropped_empty_answer": 0,
        "fixed_answer_count": 0,
    }

    for row in rows:
        r = dict(row)

        r["question"] = normalize_text(r.get("question", ""))
        r["options"] = normalize_options(r.get("options", {}))
        r["answer"] = normalize_answer(r.get("answer", []))

        if len(r["answer"]) == 0:
            stats["dropped_empty_answer"] += 1
            continue

        old_count = r.get("answer_count")
        new_count = len(r["answer"])
        if old_count != new_count:
            stats["fixed_answer_count"] += 1
        r["answer_count"] = new_count

        # Keep schema-consistent defaults.
        r["is_case-based"] = bool(r.get("is_case-based", False))
        r["is_linked"] = bool(r.get("is_linked", False))
        r["is_calc"] = bool(r.get("is_calc", False))
        if "image" not in r:
            r["image"] = None

        cleaned.append(r)

    stats["output_rows"] = len(cleaned)
    return cleaned, stats


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_infer_csv(rows: List[Dict[str, Any]], source_dataset: str = "jmedqa", lang: str = "ja") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        rec = {
            "id": str(row.get("id", "")),
            "problem_unique_id": str(row.get("problem_unique_id", "")),
            "year": str(row.get("year", "")),
            "section": str(row.get("section", "")),
            "clinical_area": str(row.get("clinical_area", "")),
            "question": str(row.get("question", "")),
            "options_json": json.dumps(row.get("options", {}), ensure_ascii=False),
            "answer_json": json.dumps(row.get("answer", []), ensure_ascii=False),
            "answer_count": str(row.get("answer_count", "")),
            "is_calc": str(bool(row.get("is_calc", False))).lower(),
            "is_linked": str(bool(row.get("is_linked", False))).lower(),
            "is_case_based": str(bool(row.get("is_case-based", False))).lower(),
            "image": "" if row.get("image") is None else str(row.get("image")),
            "source_dataset": source_dataset,
            "lang": lang,
        }
        out.append(rec)
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal cleaning for jmedqa jsonl")
    parser.add_argument("--input-jsonl", default="data/jmedqa.jsonl")
    parser.add_argument("--output-jsonl", default="data/jmedqa_clean.jsonl")
    parser.add_argument("--output-csv", default="data/jmedqa_clean.csv")
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv)

    rows = load_jsonl(in_path)
    cleaned, stats = clean_rows(rows)

    write_jsonl(out_jsonl, cleaned)
    write_csv(out_csv, to_infer_csv(cleaned))

    print("[OK] jmedqa cleaned")
    for k in ["input_rows", "dropped_empty_answer", "fixed_answer_count", "output_rows"]:
        print(f"- {k}: {stats[k]}")
    print(f"- jsonl: {out_jsonl}")
    print(f"- csv:   {out_csv}")


if __name__ == "__main__":
    main()
