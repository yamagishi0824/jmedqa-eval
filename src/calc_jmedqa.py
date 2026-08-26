#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import List

import pandas as pd


def list_csv_files(input_dir: str, pattern: str, recursive: bool) -> List[str]:
    if recursive:
        p = os.path.join(input_dir, "**", pattern)
        return sorted(glob.glob(p, recursive=True))
    p = os.path.join(input_dir, pattern)
    return sorted(glob.glob(p))


def summarize_acc(df: pd.DataFrame, keys: List[str], run_name: str, table: str) -> pd.DataFrame:
    if keys:
        g = df.groupby(keys, dropna=False, as_index=False).agg(
            n=("is_correct", "size"),
            correct=("is_correct", "sum"),
        )
    else:
        g = pd.DataFrame(
            {
                "n": [int(df["is_correct"].size)],
                "correct": [int(df["is_correct"].sum())],
            }
        )

    g["acc"] = g["correct"] / g["n"]
    g.insert(0, "run", run_name)
    g.insert(0, "table", table)
    return g


def summarize_violation_rate(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    violation_cols = [c for c in df.columns if c.startswith("violation_")]
    cols = [c for c in violation_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    block = pd.DataFrame({"metric": cols, "rate": [float(df[c].mean()) for c in cols]})
    block.insert(0, "run", run_name)
    block.insert(0, "table", "violation_rate")
    return block


def summarize_extractor_effect(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    required = {"prediction_stage1_direct", "prediction", "gold_answer"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    tmp = df.copy()
    tmp["stage1_is_correct"] = (tmp["prediction_stage1_direct"].fillna("").astype(str) == tmp["gold_answer"].fillna("").astype(str)).astype(int)
    tmp["extractor_is_correct"] = (tmp["prediction"].fillna("").astype(str) == tmp["gold_answer"].fillna("").astype(str)).astype(int)
    tmp["improved_correctness"] = ((tmp["stage1_is_correct"] == 0) & (tmp["extractor_is_correct"] == 1)).astype(int)
    tmp["degraded_correctness"] = ((tmp["stage1_is_correct"] == 1) & (tmp["extractor_is_correct"] == 0)).astype(int)
    tmp["changed_prediction"] = (tmp["prediction_stage1_direct"].fillna("").astype(str) != tmp["prediction"].fillna("").astype(str)).astype(int)

    out = pd.DataFrame(
        {
            "table": ["extractor_effect"],
            "run": [run_name],
            "n": [int(len(tmp))],
            "stage1_acc": [float(tmp["stage1_is_correct"].mean())],
            "extractor_acc": [float(tmp["extractor_is_correct"].mean())],
            "delta_acc": [float(tmp["extractor_is_correct"].mean() - tmp["stage1_is_correct"].mean())],
            "changed_prediction_rate": [float(tmp["changed_prediction"].mean())],
            "improved_correctness_rate": [float(tmp["improved_correctness"].mean())],
            "degraded_correctness_rate": [float(tmp["degraded_correctness"].mean())],
        }
    )
    return out


def paired_variant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only problem IDs having both original and no-image predictions."""
    required = {"question_variant", "is_correct"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    key = "problem_unique_id" if "problem_unique_id" in df.columns else "id"
    if key not in df.columns:
        return pd.DataFrame()

    variants = df["question_variant"].fillna("").astype(str)
    candidate = df[variants.isin(["original", "no_image"])].copy()
    counts = candidate.groupby(key)["question_variant"].nunique()
    paired_ids = counts[counts == 2].index
    return candidate[candidate[key].isin(paired_ids)].copy()


def summarize_question_variant_effect(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    paired = paired_variant_rows(df)
    if paired.empty:
        return pd.DataFrame()

    key = "problem_unique_id" if "problem_unique_id" in paired.columns else "id"
    pivot = paired.pivot_table(
        index=key,
        columns="question_variant",
        values="is_correct",
        aggfunc="first",
    ).dropna(subset=["original", "no_image"])
    if pivot.empty:
        return pd.DataFrame()

    original = pivot["original"].astype(int)
    no_image = pivot["no_image"].astype(int)
    return pd.DataFrame(
        {
            "table": ["question_variant_effect"],
            "run": [run_name],
            "n": [int(len(pivot))],
            "original_acc": [float(original.mean())],
            "no_image_acc": [float(no_image.mean())],
            "delta_acc": [float(no_image.mean() - original.mean())],
            "improved_rate": [float(((original == 0) & (no_image == 1)).mean())],
            "degraded_rate": [float(((original == 1) & (no_image == 0)).mean())],
            "unchanged_rate": [float((original == no_image).mean())],
        }
    )


def run_analysis(input_dir: str, output_file: str, pattern: str, recursive: bool) -> None:
    csv_files = list_csv_files(input_dir, pattern, recursive)
    if not csv_files:
        raise FileNotFoundError(f"No csv matched: dir={input_dir} pattern={pattern}")

    all_blocks: List[pd.DataFrame] = []

    for csv_path in csv_files:
        run_name = os.path.relpath(csv_path, input_dir)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] skip {run_name}: read failed ({e})")
            continue

        if "is_correct" not in df.columns:
            print(f"[WARN] skip {run_name}: missing is_correct")
            continue

        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)

        # Keep legacy headline metrics comparable: when both prompt variants are
        # present, the standard tables use the original exam text only.
        standard_df = df
        if "question_variant" in df.columns:
            original_df = df[df["question_variant"].fillna("").astype(str) == "original"]
            if not original_df.empty:
                standard_df = original_df

        all_blocks.append(summarize_acc(standard_df, [], run_name, "overall"))

        for col in ["year", "section", "clinical_area", "is_calc", "answer_mode", "parse_method"]:
            if col in standard_df.columns:
                all_blocks.append(summarize_acc(standard_df, [col], run_name, f"by_{col}"))

        # Combined views useful for jmedqa
        if "year" in standard_df.columns and "section" in standard_df.columns:
            all_blocks.append(summarize_acc(standard_df, ["year", "section"], run_name, "by_year_section"))

        if "year" in standard_df.columns and "clinical_area" in standard_df.columns:
            all_blocks.append(summarize_acc(standard_df, ["year", "clinical_area"], run_name, "by_year_clinical_area"))

        # Variant accuracy is calculated on the paired image-question cohort so
        # original and no-image rows always have the same denominator.
        paired = paired_variant_rows(df)
        if not paired.empty:
            all_blocks.append(summarize_acc(paired, ["question_variant"], run_name, "by_question_variant"))
            if "image_dependency" in paired.columns:
                all_blocks.append(
                    summarize_acc(
                        paired,
                        ["image_dependency", "question_variant"],
                        run_name,
                        "by_image_dependency_question_variant",
                    )
                )

        variant_effect = summarize_question_variant_effect(df, run_name)
        if not variant_effect.empty:
            all_blocks.append(variant_effect)

        v = summarize_violation_rate(standard_df, run_name)
        if not v.empty:
            all_blocks.append(v)
            if "answer_mode" in df.columns:
                mode_blocks = []
                for mode, sub_df in standard_df.groupby("answer_mode", dropna=False):
                    vv = summarize_violation_rate(sub_df, run_name)
                    if vv.empty:
                        continue
                    vv["answer_mode"] = mode
                    vv["table"] = "violation_rate_by_answer_mode"
                    mode_blocks.append(vv)
                if mode_blocks:
                    all_blocks.extend(mode_blocks)

        eff = summarize_extractor_effect(standard_df, run_name)
        if not eff.empty:
            all_blocks.append(eff)

    if not all_blocks:
        raise RuntimeError("No valid csv files were processed")

    out_df = pd.concat(all_blocks, ignore_index=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate jmedqa accuracy from prediction light csv")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--pattern", default="jmedqa_pred_light.csv")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    run_analysis(
        input_dir=args.input_dir,
        output_file=args.output_file,
        pattern=args.pattern,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
