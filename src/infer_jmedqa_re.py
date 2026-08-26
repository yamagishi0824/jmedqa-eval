#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from jmedqa_variants import QUESTION_VARIANTS, expand_question_variants

try:
    from openai_harmony import format_harmony_messages
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False

try:
    from encoding_dsv32 import encode_messages as encode_messages_dsv32
    HAS_DSV32 = True
except ImportError:
    HAS_DSV32 = False


SYSTEM_PROMPT_PRESETS: Dict[str, str] = {
    "sip": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
    "gemma": "You are a helpful medical assistant.",
    "medical": "You are an expert medical AI assistant. Answer accurately and concisely.",
}

TEMPLATE_QWEN_MED = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

TEMPLATE_LLAMA3_MED = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

MODEL_SPECIFIC_TEMPLATES = {
    "pfnet/Preferred-MedLLM-Qwen-72B": TEMPLATE_QWEN_MED,
    "pfnet/Llama3-Preferred-MedSwallow-70B": TEMPLATE_LLAMA3_MED,
}

ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:assistant\s*final\s*)?answer\s*:\s*(.+?)\s*$")


def strip_think(text: str) -> str:
    if text is None:
        return ""

    s = str(text)

    if "assistantfinal" in s:
        return s.split("assistantfinal", 1)[1].strip()

    m = re.search(r"assistant\s*final", s, flags=re.IGNORECASE)
    if m:
        return s[m.end():].strip()

    if "</think>" in s:
        return s.split("</think>", 1)[1].lstrip()

    return s.lstrip()


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def canonicalize_numeric_token(token: str) -> str:
    s = str(token).strip()
    if not s:
        return ""

    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return str(int(s))
        except Exception:
            return s

    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        try:
            x = float(s)
            if x.is_integer():
                return str(int(x))
            return ("%f" % x).rstrip("0").rstrip(".")
        except Exception:
            return s

    return s


def normalize_letters(values: List[str], allowed: List[str]) -> str:
    allowed_set = {x.lower() for x in allowed}
    picked = []
    for v in values:
        t = str(v).strip().lower()
        if len(t) != 1:
            continue
        if t not in allowed_set:
            continue
        if t in picked:
            continue
        picked.append(t)
    return ",".join(sorted(picked))


def normalize_numbers(values: List[str]) -> str:
    out = []
    for v in values:
        c = canonicalize_numeric_token(v)
        if not c:
            continue
        out.append(c)
    return ",".join(out)


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def answer_mode(row: Dict[str, Any]) -> str:
    # option: 選択肢記号解答 / numeric: 数値解答
    if not is_true(row.get("is_calc", False)):
        return "option"

    q = str(row.get("question", ""))
    opts = row.get("_options_obj", {})
    opt_vals = [str(v).strip() for v in opts.values()] if isinstance(opts, dict) else []

    if re.search(r"解答\s*[:：]", q):
        return "numeric"

    if opt_vals and all(v == "" for v in opt_vals):
        return "numeric"

    if opt_vals and len(set(opt_vals)) == 1 and "0, 1, 2" in opt_vals[0]:
        return "numeric"

    return "option"


def build_user_text(row: Dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    options = row.get("_options_obj", {})
    keys = sorted([k for k in options.keys() if isinstance(k, str)])

    n_expected = int(row.get("_answer_count_int", 1))
    mode = row.get("_answer_mode", "option")

    lines: List[str] = []

    if mode == "option":
        lines.append("次の問題について、正しい選択肢記号を答えてください。")
        if n_expected <= 1:
            lines.append("もっとも当てはまるものを1つ選んでください。")
            lines.append("最終行を1行だけ、次の形式で出力: Answer: a")
        else:
            lines.append("最終行を1行だけ、次の形式で出力: Answer: a,c")
        lines.append("選択肢記号はこの問題にある記号のみ使ってください。")
    else:
        lines.append("次の問題について、求める数値解答を答えてください。")
        if n_expected <= 1:
            lines.append("最終行を1行だけ、次の形式で出力: Answer: 40")
        else:
            lines.append("最終行を1行だけ、次の形式で出力: Answer: 12,34")
        lines.append("単位や説明は付けず、数値のみを出力してください。")

    if (
        not row.get("_suppress_image_notice", False)
        and str(row.get("image", "")).strip() != ""
    ):
        lines.append("この問題は画像参照を含みますが、画像は利用できません。問題文と選択肢にある情報のみで最善の解答をしてください。")

    lines.extend([
        "",
        "[問題]",
        question,
    ])

    if mode == "option":
        lines.extend(["", "[選択肢]"])
        for k in keys:
            lines.append(f"{k}. {str(options[k]).strip()}")

    return "\n".join(lines).strip()


def is_thinking_compatible(model_name: str) -> bool:
    """
    Chat template 側で thinking を明示できる可能性があるモデル判定。
    tokenizer実装差分があるため build_prompt() 側で TypeError fallback する。
    """
    name = model_name.lower()
    return (
        "qwen3" in name
        or "qwen3.5" in name
        or "gemma-4" in name
    )


def build_prompt(tokenizer, row: Dict[str, Any], system_prompt: str, model_name: str, enable_thinking: bool) -> str:
    is_official_gpt_oss = (
        model_name.startswith("openai/gpt-oss-120b")
        or model_name.startswith("openai/gpt-oss-20b")
    )

    current_system_prompt = system_prompt
    if is_official_gpt_oss:
        if "Reasoning:" in current_system_prompt:
            current_system_prompt = re.sub(r"Reasoning:\s*\w+", "Reasoning: medium", current_system_prompt)
        else:
            current_system_prompt = f"Reasoning: medium\n{current_system_prompt}"

    messages = [
        {"role": "system", "content": current_system_prompt},
        {"role": "user", "content": build_user_text(row)},
    ]

    if is_official_gpt_oss and HAS_HARMONY:
        return format_harmony_messages(messages)

    if "DeepSeek-V3.2" in model_name:
        if not HAS_DSV32:
            raise ImportError("encoding_dsv32.py not found. Required for DeepSeek-V3.2")
        return encode_messages_dsv32(
            messages,
            thinking_mode="thinking" if enable_thinking else "non-thinking",
            drop_thinking=False,
            add_default_bos_token=True,
        )

    kwargs: Dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
    )

    if "DeepSeek-V3" in model_name:
        kwargs["thinking"] = enable_thinking

    if is_thinking_compatible(model_name):
        kwargs["enable_thinking"] = enable_thinking
        kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        kwargs.pop("chat_template_kwargs", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)


def parse_prediction(raw_output: str, mode: str, allowed_letters: List[str]) -> str:
    text = strip_think(raw_output or "")

    m = ANSWER_LINE_RE.search(text)
    if not m:
        return ""

    body = m.group(1).strip()
    parts = re.split(r"\s*[,;]\s*", body)

    if mode == "option":
        return normalize_letters(parts, allowed_letters)

    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", body)
    return normalize_numbers(nums)


def parse_gold_answer(row: Dict[str, Any]) -> str:
    mode = row.get("_answer_mode", "option")
    ans = row.get("_answer_obj", [])

    if mode == "option":
        allowed = sorted([k for k in row.get("_options_obj", {}).keys() if isinstance(k, str)])
        return normalize_letters([str(x) for x in ans], allowed)

    return normalize_numbers([str(x) for x in ans])


def infer_expected_count(row: Dict[str, Any]) -> int:
    raw = row.get("answer_count", "")
    try:
        n = int(str(raw).strip())
        if n > 0:
            return n
    except Exception:
        pass

    ans = row.get("_answer_obj", [])
    if isinstance(ans, list) and len(ans) > 0:
        return len(ans)

    return 1


def load_csv(path: Path, question_variants: str = "both") -> List[Dict[str, Any]]:
    source_rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_rows.append(dict(row))

    rows = expand_question_variants(source_rows, question_variants)
    for r in rows:
        r["_options_obj"] = parse_json_field(r.get("options_json"), {})
        r["_answer_obj"] = parse_json_field(r.get("answer_json"), [])
        r["_answer_count_int"] = infer_expected_count(r)
        r["_answer_mode"] = answer_mode(r)
    return rows


def make_light_record(row: Dict[str, Any], prediction: str, raw_output_file: str) -> Dict[str, Any]:
    rec = {k: v for k, v in row.items() if not k.startswith("_")}
    gold = parse_gold_answer(row)

    rec["answer_mode"] = row.get("_answer_mode", "option")
    rec["expected_answer_count"] = row.get("_answer_count_int", 1)
    rec["gold_answer"] = gold
    rec["prediction"] = prediction
    rec["is_correct"] = int(prediction == gold)
    rec["raw_output_file"] = raw_output_file
    rec["parse_method"] = "regex"
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="jmedqa inference + regex extraction")

    parser.add_argument("--model", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-full", required=True)
    parser.add_argument("--out-light", required=True)
    parser.add_argument(
        "--question-variants",
        choices=QUESTION_VARIANTS,
        default="both",
        help="Prompt variants: original, no_image, or both (no_image only when question differs)",
    )

    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.90)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)

    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--system-prompt-preset", default="medical")
    parser.add_argument("--system-prompt", type=str, default=None)

    parser.add_argument("--keep-think", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")

    args = parser.parse_args()

    system_prompt = args.system_prompt or SYSTEM_PROMPT_PRESETS.get(
        args.system_prompt_preset,
        SYSTEM_PROMPT_PRESETS["medical"],
    )

    in_path = Path(args.input_csv)
    out_full = Path(args.out_full)
    out_light = Path(args.out_light)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_light.parent.mkdir(parents=True, exist_ok=True)

    rows = load_csv(in_path, args.question_variants)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.chat_template is None:
        if args.model in MODEL_SPECIFIC_TEMPLATES:
            tokenizer.chat_template = MODEL_SPECIFIC_TEMPLATES[args.model]
        else:
            tokenizer.chat_template = TEMPLATE_QWEN_MED

    prompts = [
        build_prompt(
            tokenizer=tokenizer,
            row=row,
            system_prompt=system_prompt,
            model_name=args.model,
            enable_thinking=args.enable_thinking,
        )
        for row in rows
    ]

    vllm_kwargs: Dict[str, Any] = {
        "model": args.model,
        "dtype": "auto",
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_mem,
        "max_model_len": args.max_len,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": True,
    }
    if "DeepSeek-V3.2" in args.model:
        vllm_kwargs["tokenizer_mode"] = "deepseek_v32"

    llm = LLM(**vllm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_k=args.top_k if args.top_k > 0 else -1,
    )

    def run_generate(batch_prompts: List[str]) -> List[str]:
        outs = llm.generate(batch_prompts, sampling)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    raw_results: List[str] = []
    if args.batch_size > 0:
        for start in range(0, len(prompts), args.batch_size):
            raw_results.extend(run_generate(prompts[start:start + args.batch_size]))
    else:
        raw_results = run_generate(prompts)

    light_records: List[Dict[str, Any]] = []
    raw_ref = str(out_full)

    with out_full.open("w", encoding="utf-8") as f_full:
        for row, raw in zip(rows, raw_results):
            text = raw if args.keep_think else strip_think(raw)
            allowed = sorted([k for k in row.get("_options_obj", {}).keys() if isinstance(k, str)])
            pred = parse_prediction(text, row.get("_answer_mode", "option"), allowed)

            full_rec = {k: v for k, v in row.items() if not k.startswith("_")}
            full_rec["answer_mode"] = row.get("_answer_mode", "option")
            full_rec["expected_answer_count"] = row.get("_answer_count_int", 1)
            full_rec["prediction"] = pred
            full_rec["raw_output"] = text
            f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")

            light_records.append(make_light_record(row=row, prediction=pred, raw_output_file=raw_ref))

    fieldnames = list(light_records[0].keys()) if light_records else []
    with out_light.open("w", encoding="utf-8", newline="") as f_light:
        writer = csv.DictWriter(f_light, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(light_records)

    print(f"DONE full={out_full} light={out_light} rows={len(light_records)}")


if __name__ == "__main__":
    main()
