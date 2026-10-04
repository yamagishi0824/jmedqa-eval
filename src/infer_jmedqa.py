#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JMedQA stage1: generate free-form answers with each model's official settings (src/model_profiles.py).

- System prompt, thinking switch and sampling come from the model profile.
- The prompt only specifies the answer format (`Answer: X` on the last line, prompt_style=answer_line);
  it neither encourages nor suppresses reasoning. The answer is extracted by rules (src/answer_parser.py).
- answer_mode is taken from the CSV column (questions without options are numeric).
- Sampling: --sampling greedy (temperature 0) or --sampling official (the card's values).
  The seed is fixed per question (seed = base_seed + row index).
- Prompts are built with LLM.chat, so models without a Jinja template (e.g. DeepSeek) are handled by vLLM.
- The reasoning part is split off by its delimiter; raw_output keeps only the final answer, which is scored.
  Reasoning, full output and prompt go to <out-full>_reasoning.jsonl.
- --with-images: vision evaluation. Only the original rows that reference images are inferred, with the
  original question text (question_raw) and the images (question_variant=original_image).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from answer_parser import parse_answer
from jmedqa_variants import QUESTION_VARIANTS
from model_profiles import get_profile

PROMPT_STYLES = ("answer_line", "legacy_ja", "qwen_answer", "medgemma_final")

# Added to original image-referencing questions in the text-only setting.
IMAGE_NOTICE = "この問題は画像参照を含みますが、画像は利用できません。問題文と選択肢にある情報のみで最善の解答をしてください。"

# answer_line: specify the answer format only (the last line "Answer: X"); no instruction about reasoning.
ANSWER_LINE_OPTION_SINGLE = "次の問題に答えてください。回答の最後の行に、選んだ選択肢の記号を「Answer: X」の形式で書いてください（例: Answer: c）。"
ANSWER_LINE_OPTION_MULTI = (
    "次の問題に答えてください。回答の最後の行に、問題文で指定された数の選択肢の記号を"
    "「Answer: X, Y」の形式で書いてください（例: Answer: a, c）。"
)
ANSWER_LINE_NUMERIC = "次の問題に答えてください。回答の最後の行に、求めた数値だけを「Answer: X」の形式で書いてください（単位は付けない。例: Answer: 26）。"

QWEN_ANSWER_SINGLE = 'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "c"`.'
QWEN_ANSWER_MULTI = (
    "Please show your choices in the `answer` field with only the choice letters separated by commas, "
    'e.g., `"answer": "a,c"`.'
)
QWEN_ANSWER_NUMERIC = 'Please show your final answer in the `answer` field with only the number, e.g., `"answer": "40"`.'

MEDGEMMA_FINAL = (
    "You may write out your argument before stating your final, very short, definitive, and concise answer "
    "(no more than a few words or the letter corresponding to your answer choice if the question is multiple choice) "
    "X in the format 'Final Answer: X':"
)


def _question_block(row: Dict[str, Any]) -> List[str]:
    question = str(row.get("question", "")).strip()
    options = row.get("_options_obj", {})
    lines: List[str] = []
    if not row.get("_suppress_image_notice", False) and str(row.get("image", "")).strip() != "":
        lines.extend([IMAGE_NOTICE, ""])
    lines.extend(["[問題]", question])
    if row.get("_answer_mode", "option") == "option":
        lines.extend(["", "[選択肢]"])
        for k in sorted(k for k in options.keys() if isinstance(k, str)):
            lines.append(f"{k}. {str(options[k]).strip()}")
    return lines


def build_user_text(row: Dict[str, Any], style: str) -> str:
    if style == "legacy_ja":
        from infer_jmedqa_extract_llm import build_user_text as legacy_build_user_text
        return legacy_build_user_text(row)

    lines = _question_block(row)
    mode = row.get("_answer_mode", "option")
    if style == "answer_line":
        if mode != "option":
            head = ANSWER_LINE_NUMERIC
        elif int(row.get("_answer_count_int", 1)) > 1:
            head = ANSWER_LINE_OPTION_MULTI
        else:
            head = ANSWER_LINE_OPTION_SINGLE
        return "\n".join([head, ""] + lines).strip()
    if style == "qwen_answer":
        if mode != "option":
            fmt = QWEN_ANSWER_NUMERIC
        elif int(row.get("_answer_count_int", 1)) > 1:
            fmt = QWEN_ANSWER_MULTI
        else:
            fmt = QWEN_ANSWER_SINGLE
        lines.extend(["", fmt])
    elif style == "medgemma_final":
        lines.append(MEDGEMMA_FINAL)
    else:
        raise ValueError(f"unknown prompt_style: {style}")
    return "\n".join(lines).strip()


# Reasoning delimiters: (open tag, close tag). The open tag may be part of the prompt and absent from the output.
REASONING_TAGS = {
    "gemma4": ("<|channel>", "<channel|>"),
    "medgemma": ("<unused94>", "<unused95>"),
    "think_tag": ("<think>", "</think>"),
    "harmony": ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),  # gpt-oss / llm-jp-4
}


def split_reasoning(text: str, kind: str, thinking_on: bool) -> Tuple[str, str, bool]:
    """Return (reasoning, response, reasoning_closed).

    - With a close tag: text before the last close tag is reasoning, the rest is the response.
    - Only an open tag (truncated), or thinking on without a close tag: everything is reasoning, response is empty.
    - No tags and thinking off: everything is the response.
    """
    open_tag, close_tag = REASONING_TAGS[kind]
    if close_tag in text:
        head, tail = text.rsplit(close_tag, 1)
        return head.replace(open_tag, "", 1).strip(), tail.strip(), True
    if open_tag in text:
        return text.replace(open_tag, "", 1).strip(), "", False
    if thinking_on and kind in ("think_tag", "harmony"):
        return text.strip(), "", False
    return "", text.strip(), True


def strip_special_tokens(text: str, special_tokens: List[str]) -> str:
    for t in special_tokens:
        if t:
            text = text.replace(t, "")
    return text.strip()


def main() -> None:
    p = argparse.ArgumentParser(description="JMedQA stage1 (official per-model settings)")
    p.add_argument("--model", required=True, help="local checkpoint directory or Hugging Face repo id")
    p.add_argument("--profile", required=True)
    p.add_argument("--mode", choices=["think", "nothink"], required=True)
    p.add_argument("--sampling", choices=["greedy", "official"], default="greedy")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--out-full", required=True)
    p.add_argument("--out-light", required=True)
    p.add_argument("--question-variants", choices=QUESTION_VARIANTS, default="both")
    p.add_argument("--prompt-style", choices=PROMPT_STYLES, default=None, help="override the profile's prompt_style")
    p.add_argument("--chat-template-kwargs", default=None,
                   help='JSON merged into the profile\'s chat_template_kwargs (e.g. {"reasoning_effort": "high"})')
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=0, help="0 = profile value")
    p.add_argument("--max-tokens", type=int, default=0, help="0 = profile value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--with-images", default=None,
                   help="vision evaluation: directory holding the dataset's images/ (a local copy of SIP-med-LLM/JMedQA)")
    args = p.parse_args()

    # vLLM re-imports this module in spawned workers, so heavy imports stay inside main().
    # Dataset loading and record helpers are shared with the v1 pipeline (src/infer_jmedqa_extract_llm.py).
    from infer_jmedqa_extract_llm import analyze_extraction, load_csv, make_light_record
    from vllm import LLM, SamplingParams

    prof = get_profile(args.profile, args.mode, args.sampling)
    if args.prompt_style:
        prof["prompt_style"] = args.prompt_style
    max_model_len = args.max_model_len or prof["max_model_len"]
    max_tokens = args.max_tokens or prof["max_tokens"]
    ctk = dict(prof["chat_template_kwargs"])
    if args.chat_template_kwargs:
        ctk.update(json.loads(args.chat_template_kwargs))
        prof["chat_template_kwargs"] = dict(ctk)  # recorded in run_config.json
    thinking_on = args.mode == "think"

    rows = load_csv(Path(args.input_csv), args.question_variants)
    for r in rows:  # prefer the CSV's answer_mode column
        if str(r.get("answer_mode", "")).strip() in ("option", "numeric"):
            r["_answer_mode"] = str(r["answer_mode"]).strip()
    image_root = None
    if args.with_images:
        image_root = Path(args.with_images).resolve()
        rows = [r for r in rows if r.get("question_variant") == "original" and json.loads(r.get("image_paths_json") or "{}")]
        for r in rows:
            r["question_variant"] = "original_image"
            r["_suppress_image_notice"] = True  # images are provided, so no "image unavailable" notice
    if args.limit > 0:
        rows = rows[: args.limit]

    def image_content(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Images placed before the question, each labelled with its reference name (order of image_paths_json)."""
        parts: List[Dict[str, Any]] = []
        for ref, paths in json.loads(row.get("image_paths_json") or "{}").items():
            for rel in paths:
                f = image_root / rel
                if not f.is_file():
                    raise FileNotFoundError(f"missing image: {f}")
                parts.append({"type": "text", "text": f"[{ref}]"})
                parts.append({"type": "image_url", "image_url": {"url": f.as_uri()}})
        return parts

    conversations = []
    for row in rows:
        msgs = []
        if prof["system_prompt"] is not None:
            msgs.append({"role": "system", "content": prof["system_prompt"]})
        if image_root is not None:
            msgs.append({"role": "user", "content": image_content(row) + [{"type": "text", "text": build_user_text(row, prof["prompt_style"])}]})
        else:
            msgs.append({"role": "user", "content": build_user_text(row, prof["prompt_style"])})
        conversations.append(msgs)

    out_full = Path(args.out_full)
    out_light = Path(args.out_light)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_light.parent.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model": args.model, "profile": args.profile, "mode": args.mode, "profile_settings": prof,
        "max_model_len": max_model_len, "max_tokens": max_tokens, "tp": args.tp, "seed": args.seed,
        "question_variants": args.question_variants, "n_rows": len(rows),
        "with_images": bool(image_root),
    }
    (out_full.parent / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[CONFIG]", json.dumps(run_config, ensure_ascii=False))

    llm_kwargs: Dict[str, Any] = {
        "model": args.model,
        "dtype": "auto",
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_mem,
        "max_model_len": max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "safetensors_prefetch_num_threads": 16,
    }
    llm_kwargs.update(prof["llm_kwargs"])
    if image_root is not None:
        max_imgs = max(sum(len(v) for v in json.loads(r.get("image_paths_json") or "{}").values()) for r in rows)
        llm_kwargs["limit_mm_per_prompt"] = {"image": max_imgs}
        llm_kwargs["allowed_local_media_path"] = str(image_root)
    llm = LLM(**llm_kwargs)

    tok = llm.get_tokenizer()
    special_tokens = sorted(set(getattr(tok, "all_special_tokens", []) or []), key=len, reverse=True)

    per_row_max = [max_tokens] * len(rows)
    if prof.get("clip_max_tokens_to_context"):  # short-context models: max_tokens = context - prompt length
        for i, msgs in enumerate(conversations):
            prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  chat_template=prof.get("chat_template"), **ctk)
            n_prompt = len(tok.encode(prompt_text, add_special_tokens=False))
            per_row_max[i] = max(1, min(max_tokens, max_model_len - n_prompt))
        print(f"[INFO] max_tokens clipped to the context: min={min(per_row_max)} max={max(per_row_max)}")
    sampling = [
        SamplingParams(max_tokens=per_row_max[i], seed=args.seed + i, skip_special_tokens=False, **prof["sampling"])
        for i in range(len(rows))
    ]
    outs = llm.chat(conversations, sampling, chat_template=prof.get("chat_template"), chat_template_kwargs=ctk, use_tqdm=True)

    light_records: List[Dict[str, Any]] = []
    n_trunc = 0
    out_reasoning = out_full.with_name(out_full.stem + "_reasoning.jsonl")
    with out_full.open("w", encoding="utf-8") as f_full, out_reasoning.open("w", encoding="utf-8") as f_rsn:
        for row, o in zip(rows, outs):
            c = o.outputs[0] if o.outputs else None
            if c and prof.get("redecode"):
                text = tok.decode(list(c.token_ids), skip_special_tokens=False)
            else:
                text = c.text if c else ""
            finish = c.finish_reason if c else "none"
            n_tok = len(c.token_ids) if c else 0

            reasoning, response, closed = split_reasoning(text, prof["reasoning"], thinking_on)
            reasoning = strip_special_tokens(reasoning, special_tokens)
            response = strip_special_tokens(response, special_tokens)
            # Finished normally ("stop") without the final-answer delimiter: the model answered inside the
            # reasoning channel. It is not a truncation, so the reasoning text is treated as the response.
            if not closed and finish == "stop" and not response:
                response = reasoning
                reasoning = ""
            # Token counts of the reasoning and response parts (delimiters excluded), with the model's tokenizer
            n_rsn_tok = len(tok.encode(reasoning, add_special_tokens=False)) if reasoning else 0
            n_cnt_tok = len(tok.encode(response, add_special_tokens=False)) if response else 0
            if finish == "length":
                n_trunc += 1

            options = {k: v for k, v in row.get("_options_obj", {}).items() if isinstance(k, str)}
            row_mode = row.get("_answer_mode", "option")
            pred_direct, rule_method = parse_answer(response, row_mode, sorted(options), options)

            full_rec = {k: v for k, v in row.items() if not k.startswith("_")}
            full_rec["answer_mode"] = row_mode
            full_rec["expected_answer_count"] = row.get("_answer_count_int", 1)
            full_rec["prediction_stage1_direct"] = pred_direct
            full_rec["prediction_rule"] = pred_direct
            full_rec["rule_method"] = rule_method
            full_rec["raw_output"] = response  # final answer part (reasoning removed)
            full_rec["reasoning_closed"] = int(closed)
            full_rec["reasoning_chars"] = len(reasoning)
            full_rec["n_reasoning_tokens"] = n_rsn_tok
            full_rec["n_content_tokens"] = n_cnt_tok
            full_rec["finish_reason"] = finish
            full_rec["n_output_tokens"] = n_tok
            f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")
            f_rsn.write(json.dumps({
                "id": row.get("id"), "question_variant": row.get("question_variant"),
                "prompt": o.prompt, "reasoning": reasoning, "raw_output_full": text,
            }, ensure_ascii=False) + "\n")

            flags = analyze_extraction(row, pred_direct, pred_direct)
            rec = make_light_record(row=row, prediction=pred_direct, raw_output_file=str(out_full),
                                    stage1_direct=pred_direct, extraction_flags=flags)
            rec["parse_method"] = "rule"
            rec["rule_method"] = rule_method
            rec["finish_reason"] = finish
            rec["n_output_tokens"] = n_tok
            rec["n_reasoning_tokens"] = n_rsn_tok
            rec["n_content_tokens"] = n_cnt_tok
            light_records.append(rec)

    fieldnames = list(light_records[0].keys()) if light_records else []
    with out_light.open("w", encoding="utf-8", newline="") as f_light:
        writer = csv.DictWriter(f_light, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(light_records)
    n = max(len(light_records), 1)
    acc = sum(r["is_correct"] for r in light_records) / n
    fmt = sum(r["rule_method"] == "answer_line" for r in light_records) / n
    print(f"DONE_STAGE1 full={out_full} light={out_light} rows={len(light_records)} truncated={n_trunc} "
          f"rule_acc={acc:.4f} answer_line_rate={fmt:.4f}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)  # make sure vLLM engine processes do not keep holding the GPUs after a failure
