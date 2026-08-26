#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List


QUESTION_VARIANTS = ("both", "original", "no_image")


def expand_question_variants(
    source_rows: Iterable[Dict[str, Any]],
    selection: str = "both",
) -> List[Dict[str, Any]]:
    """Create prompt rows for the original and image-reference-removed text.

    ``question_raw`` is the original exam text and ``question`` is the text with
    image references removed.  A no-image row is emitted only when those fields
    differ, so ordinary text-only questions are never inferred twice.
    """
    if selection not in QUESTION_VARIANTS:
        raise ValueError(f"Unknown question variant: {selection}")

    expanded: List[Dict[str, Any]] = []
    for source in source_rows:
        original_question = str(source.get("question_raw") or source.get("question") or "")
        no_image_question = str(source.get("question") or "")
        has_no_image_variant = bool(original_question and original_question != no_image_question)

        if selection in {"both", "original"}:
            row = dict(source)
            row["question"] = original_question
            row["question_variant"] = "original"
            row["has_no_image_variant"] = has_no_image_variant
            row["_suppress_image_notice"] = False
            expanded.append(row)

        if selection in {"both", "no_image"} and has_no_image_variant:
            row = dict(source)
            row["question"] = no_image_question
            row["question_variant"] = "no_image"
            row["has_no_image_variant"] = True
            # The prompt text no longer refers to an image, so do not append the
            # legacy "this question references an image" instruction.
            row["_suppress_image_notice"] = True
            expanded.append(row)

    return expanded
