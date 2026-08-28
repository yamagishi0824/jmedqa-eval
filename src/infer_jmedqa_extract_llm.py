#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

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

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a strict answer extraction engine. "
    "Your task is to read only the given model output and extract the answer that the model presents as its final answer. "
    "Do not solve the question yourself. "
    "Do not use medical knowledge. "
    "Do not infer an answer that is not stated or clearly presented by the model output. "
    "Do not collect option letters or numbers that are merely mentioned in reasoning, explanation, comparison, negation, or elimination. "
    "Extract only the option letter(s) or numeric value(s) that the model output appears to select as the answer. "
    "If the model output contains an explicit final-answer expression, prioritize that expression over earlier reasoning. "
    "Output only the extracted answer value, with no prefix/suffix. "
    "If no valid final answer exists in the text, output: EMPTY"
)

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
    picked: List[str] = []
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
        if c:
            out.append(c)
    return ",".join(out)


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def answer_mode(row: Dict[str, Any]) -> str:
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
        lines.append("回答形式は自由です。")
    else:
        lines.append("次の問題について、求める数値解答を答えてください。")
        lines.append("回答形式は自由です。")

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


def build_extractor_prompt(tokenizer, raw_output: str, mode: str) -> str:
    # 問題文や正答数は渡さない。モデル出力のみを入力に使う。
    # 目的は「出力中に登場した全選択肢」ではなく、
    # 「モデルが答案として提示しているもの」を抽出すること。
    if mode == "option":
        mode_instruction = (
            "Target answer type: option letter(s).\n"
            "Return only the option letter or option letters that the model output presents as its answer.\n"
            "Valid examples of the output format are: a or a,c.\n"
            "Do not output option letters that are only mentioned while explaining, comparing, rejecting, or listing alternatives.\n"
            "If the model clearly gives one final answer, output only that one answer even if other option letters appear elsewhere.\n"
            "If the model clearly gives multiple final answers, output those final answers separated by commas.\n"
        )
    else:
        mode_instruction = (
            "Target answer type: numeric value(s).\n"
            "Return only the numeric value or numeric values that the model output presents as its answer.\n"
            "Valid examples of the output format are: 40 or 12,34.\n"
            "Do not output numbers that are only mentioned while explaining, calculating intermediate steps, comparing, or rejecting alternatives.\n"
            "If the model clearly gives one final numeric answer, output only that one answer even if other numbers appear elsewhere.\n"
            "If the model clearly gives multiple final numeric answers, output those final answers separated by commas.\n"
        )

    user_text = (
        "[Model Output]\n"
        f"{raw_output}\n\n"
        "[Extraction Task]\n"
        "Extract the answer that the model output itself appears to present as the answer.\n"
        "Do not judge whether the answer is medically correct.\n"
        "Do not change the answer to match the expected number of answers.\n"
        "Do not collect every option letter or number that appears in the text.\n"
        "Prioritize expressions that indicate a final answer, such as '正解は', '答えは', '解答は', "
        "'したがって', 'よって', 'therefore', 'answer', or 'final answer'.\n"
        "If later text clearly revises or overrides an earlier answer, use the later final answer.\n"
        "If no answer is stated or clearly presented, output EMPTY.\n\n"
        f"{mode_instruction}\n"
        "[Output Format]\n"
        "Output only the extracted answer value. No explanation. No prefix. No suffix."
    )
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_prediction_from_text(text: str, mode: str, allowed_letters: List[str]) -> str:
    body = strip_think(text or "").strip()
    if not body:
        return ""

    m = ANSWER_LINE_RE.search(body)
    if m:
        body = m.group(1).strip()

    if body.strip().upper() == "EMPTY":
        return ""

    parts = re.split(r"[\s,;、/]+", body)

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


def count_answer_items(pred: str) -> int:
    if not pred:
        return 0
    return len([x for x in pred.split(",") if x.strip()])


def analyze_extraction(row: Dict[str, Any], stage1_direct: str, extracted: str) -> Dict[str, Any]:
    mode = row.get("_answer_mode", "option")
    expected_n = int(row.get("_answer_count_int", 1))

    out: Dict[str, Any] = {}

    stage1_n = count_answer_items(stage1_direct)
    extracted_n = count_answer_items(extracted)

    out["stage1_answer_items"] = stage1_n
    out["extractor_answer_items"] = extracted_n

    out["violation_stage1_no_parse"] = int(stage1_direct == "")
    out["violation_extractor_no_parse"] = int(extracted == "")

    out["violation_stage1_wrong_answer_count"] = int(stage1_n > 0 and stage1_n != expected_n)
    out["violation_wrong_answer_count"] = int(extracted_n > 0 and extracted_n != expected_n)

    out["extractor_changed_prediction"] = int(stage1_direct != extracted)
    out["extractor_improved_parse"] = int(stage1_direct == "" and extracted != "")
    out["extractor_degraded_parse"] = int(stage1_direct != "" and extracted == "")

    if mode == "option":
        stage1_tokens = [x.strip().lower() for x in stage1_direct.split(",") if x.strip()]
        tokens = [x.strip().lower() for x in extracted.split(",") if x.strip()]
        out["violation_stage1_non_option_answer"] = int(any(not re.fullmatch(r"[a-z]", t) for t in stage1_tokens))
        out["violation_non_option_answer"] = int(any(not re.fullmatch(r"[a-z]", t) for t in tokens))
        out["violation_stage1_non_numeric_answer"] = 0
        out["violation_non_numeric_answer"] = 0
    else:
        stage1_tokens = [x.strip() for x in stage1_direct.split(",") if x.strip()]
        tokens = [x.strip() for x in extracted.split(",") if x.strip()]
        out["violation_stage1_non_numeric_answer"] = int(any(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t) is None for t in stage1_tokens))
        out["violation_non_numeric_answer"] = int(any(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t) is None for t in tokens))
        out["violation_stage1_non_option_answer"] = 0
        out["violation_non_option_answer"] = 0

    return out


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


def enrich_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["_options_obj"] = parse_json_field(r.get("options_json"), {})
        r["_answer_obj"] = parse_json_field(r.get("answer_json"), [])
        r["_answer_count_int"] = infer_expected_count(r)
        r["_answer_mode"] = answer_mode(r)
        out.append(r)
    return out


def make_light_record(
    row: Dict[str, Any],
    prediction: str,
    raw_output_file: str,
    stage1_direct: str,
    extraction_flags: Dict[str, Any],
) -> Dict[str, Any]:
    rec = {k: v for k, v in row.items() if not k.startswith("_")}
    gold = parse_gold_answer(row)

    rec["answer_mode"] = row.get("_answer_mode", "option")
    rec["expected_answer_count"] = row.get("_answer_count_int", 1)
    rec["gold_answer"] = gold
    rec["prediction"] = prediction
    rec["prediction_stage1_direct"] = stage1_direct
    rec["is_correct"] = int(prediction == gold)
    rec["raw_output_file"] = raw_output_file
    rec["parse_method"] = "llm_extractor"

    for k, v in extraction_flags.items():
        rec[k] = v

    return rec


def load_extract_jobs(path: Path) -> List[Dict[str, Path]]:
    """Read an extraction job list (TSV: stage1_full_in<TAB>out_full<TAB>out_light)."""
    jobs: List[Dict[str, Path]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"invalid jobs line (need 3 tab-separated fields): {path}:{line_no}: {line}"
                )
            jobs.append(
                {
                    "stage1_in": Path(parts[0]),
                    "out_full": Path(parts[1]),
                    "out_light": Path(parts[2]),
                }
            )
    return jobs


def run_extract_job(
    extractor_llm: LLM,
    extractor_tokenizer: Any,
    extractor_sampling: SamplingParams,
    generate_fn,
    stage1_in: Path,
    out_full: Path,
    out_light: Path,
    extractor_batch_size: int = 0,
) -> int:
    """Run extraction for one stage1 file using an already loaded extractor LLM."""
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_light.parent.mkdir(parents=True, exist_ok=True)

    rows = enrich_rows(load_jsonl(stage1_in))

    stage1_raw = [str(r.get("raw_output", "") or "") for r in rows]
    extractor_prompts = [
        build_extractor_prompt(extractor_tokenizer, raw, row.get("_answer_mode", "option"))
        for raw, row in zip(stage1_raw, rows)
    ]

    extractor_raw: List[str] = []
    if extractor_batch_size > 0:
        for start in range(0, len(extractor_prompts), extractor_batch_size):
            extractor_raw.extend(
                generate_fn(
                    extractor_llm,
                    extractor_prompts[start:start + extractor_batch_size],
                    extractor_sampling,
                )
            )
    else:
        extractor_raw = generate_fn(extractor_llm, extractor_prompts, extractor_sampling)

    light_records: List[Dict[str, Any]] = []
    raw_ref = str(out_full)
    with out_full.open("w", encoding="utf-8") as f_full:
        for row, raw1, raw2 in zip(rows, stage1_raw, extractor_raw):
            text1 = str(raw1)
            text2 = strip_think(raw2)

            allowed = sorted([k for k in row.get("_options_obj", {}).keys() if isinstance(k, str)])
            row_mode = row.get("_answer_mode", "option")
            pred_stage1_direct = parse_prediction_from_text(text1, row_mode, allowed)
            pred_extracted = parse_prediction_from_text(text2, row_mode, allowed)
            flags = analyze_extraction(row, pred_stage1_direct, pred_extracted)

            full_rec = {k: v for k, v in row.items() if not k.startswith("_")}
            full_rec["answer_mode"] = row_mode
            full_rec["expected_answer_count"] = row.get("_answer_count_int", 1)
            full_rec["prediction_stage1_direct"] = pred_stage1_direct
            full_rec["prediction"] = pred_extracted
            full_rec["extractor_output"] = text2
            for k, v in flags.items():
                full_rec[k] = v
            f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")

            light_records.append(
                make_light_record(
                    row=row,
                    prediction=pred_extracted,
                    raw_output_file=raw_ref,
                    stage1_direct=pred_stage1_direct,
                    extraction_flags=flags,
                )
            )

    fieldnames = list(light_records[0].keys()) if light_records else []
    with out_light.open("w", encoding="utf-8", newline="") as f_light:
        writer = csv.DictWriter(f_light, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(light_records)
    return len(light_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="jmedqa inference + LLM extraction")

    parser.add_argument("--mode", choices=["all", "stage1", "extract"], default="all")
    parser.add_argument("--model", required=True)
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--stage1-full-in", type=str, default=None)
    parser.add_argument("--out-full", default=None)
    parser.add_argument("--out-light", default=None)
    parser.add_argument(
        "--jobs-file",
        type=str,
        default=None,
        help=(
            "extract mode only. TSV file, one job per line: "
            "stage1_full_in<TAB>out_full<TAB>out_light. "
            "All jobs are processed with a single extractor model load."
        ),
    )
    parser.add_argument(
        "--question-variants",
        choices=QUESTION_VARIANTS,
        default="both",
        help="Prompt variants used in stage1: original, no_image, or both",
    )

    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--extractor-tp", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--extractor-max-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--extractor-gpu-mem", type=float, default=0.90)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)

    parser.add_argument("--extractor-temperature", type=float, default=0.0)
    parser.add_argument("--extractor-top-p", type=float, default=1.0)
    parser.add_argument("--extractor-top-k", type=int, default=-1)
    parser.add_argument("--extractor-max-tokens", type=int, default=256)

    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--extractor-batch-size", type=int, default=0)
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

    if args.jobs_file is not None and args.mode != "extract":
        parser.error("--jobs-file is only supported with --mode extract")

    out_full = None
    out_light = None
    if args.jobs_file is None:
        if not args.out_full or not args.out_light:
            parser.error("--out-full and --out-light are required unless --jobs-file is given")
        out_full = Path(args.out_full)
        out_light = Path(args.out_light)
        out_full.parent.mkdir(parents=True, exist_ok=True)
        out_light.parent.mkdir(parents=True, exist_ok=True)

    def run_generate(model: LLM, batch_prompts: List[str], sp: SamplingParams) -> List[str]:
        outs = model.generate(batch_prompts, sp)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_k=args.top_k if args.top_k > 0 else -1,
    )
    extractor_sampling = SamplingParams(
        temperature=args.extractor_temperature,
        top_p=args.extractor_top_p,
        max_tokens=args.extractor_max_tokens,
        top_k=args.extractor_top_k if args.extractor_top_k > 0 else -1,
    )

    mode = args.mode
    if mode == "all":
        raise ValueError(
            "--mode all is disabled to avoid loading generation/extractor models simultaneously. "
            "Use --mode stage1 first, then --mode extract with --stage1-full-in."
        )

    if mode == "stage1":
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

        llm_kwargs: Dict[str, Any] = {
            "model": args.model,
            "dtype": "auto",
            "tensor_parallel_size": args.tp,
            "gpu_memory_utilization": args.gpu_mem,
            "max_model_len": args.max_len,
            "trust_remote_code": args.trust_remote_code,
            "enforce_eager": True,
        }
        if "DeepSeek-V3.2" in args.model:
            llm_kwargs["tokenizer_mode"] = "deepseek_v32"
        llm = LLM(**llm_kwargs)

        stage1_raw: List[str] = []
        if args.batch_size > 0:
            for start in range(0, len(prompts), args.batch_size):
                stage1_raw.extend(run_generate(llm, prompts[start:start + args.batch_size], sampling))
        else:
            stage1_raw = run_generate(llm, prompts, sampling)

        light_records: List[Dict[str, Any]] = []
        raw_ref = str(out_full)
        with out_full.open("w", encoding="utf-8") as f_full:
            for row, raw1 in zip(rows, stage1_raw):
                text1 = raw1 if args.keep_think else strip_think(raw1)
                allowed = sorted([k for k in row.get("_options_obj", {}).keys() if isinstance(k, str)])
                row_mode = row.get("_answer_mode", "option")
                pred_stage1_direct = parse_prediction_from_text(text1, row_mode, allowed)

                full_rec = {k: v for k, v in row.items() if not k.startswith("_")}
                full_rec["answer_mode"] = row_mode
                full_rec["expected_answer_count"] = row.get("_answer_count_int", 1)
                full_rec["prediction_stage1_direct"] = pred_stage1_direct
                full_rec["raw_output"] = text1
                f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")

                flags = analyze_extraction(row, pred_stage1_direct, pred_stage1_direct)
                light_records.append(
                    make_light_record(
                        row=row,
                        prediction=pred_stage1_direct,
                        raw_output_file=raw_ref,
                        stage1_direct=pred_stage1_direct,
                        extraction_flags=flags,
                    )
                )

        fieldnames = list(light_records[0].keys()) if light_records else []
        with out_light.open("w", encoding="utf-8", newline="") as f_light:
            writer = csv.DictWriter(f_light, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(light_records)
        print(f"DONE_STAGE1 full={out_full} light={out_light} rows={len(light_records)}")
        return

    # mode == "extract"
    if args.jobs_file:
        raw_jobs = load_extract_jobs(Path(args.jobs_file))
    else:
        stage1_in = Path(args.stage1_full_in) if args.stage1_full_in else out_full
        if not stage1_in.exists():
            raise FileNotFoundError(f"stage1 full jsonl not found: {stage1_in}")
        raw_jobs = [{"stage1_in": stage1_in, "out_full": out_full, "out_light": out_light}]

    # Skip jobs without stage1 input and jobs whose outputs already exist.
    jobs: List[Dict[str, Path]] = []
    for job in raw_jobs:
        if not job["stage1_in"].exists():
            print(f"SKIP_EXTRACT reason=stage1_missing stage1={job['stage1_in']}")
            continue
        if job["out_full"].exists() and job["out_light"].exists():
            print(f"SKIP_EXTRACT reason=already_done full={job['out_full']} light={job['out_light']}")
            continue
        jobs.append(job)

    if not jobs:
        print("DONE_EXTRACT jobs=0 (nothing to extract)")
        return

    extractor_tokenizer = AutoTokenizer.from_pretrained(
        args.extractor_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if extractor_tokenizer.chat_template is None:
        if args.extractor_model in MODEL_SPECIFIC_TEMPLATES:
            extractor_tokenizer.chat_template = MODEL_SPECIFIC_TEMPLATES[args.extractor_model]
        else:
            extractor_tokenizer.chat_template = TEMPLATE_QWEN_MED

    extractor_llm_kwargs: Dict[str, Any] = {
        "model": args.extractor_model,
        "dtype": "auto",
        "tensor_parallel_size": args.extractor_tp,
        "gpu_memory_utilization": args.extractor_gpu_mem,
        "max_model_len": args.extractor_max_len,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": True,
    }
    if "DeepSeek-V3.2" in args.extractor_model:
        extractor_llm_kwargs["tokenizer_mode"] = "deepseek_v32"
    # The extractor model is loaded once and reused for every job.
    extractor_llm = LLM(**extractor_llm_kwargs)

    n_ok = 0
    n_failed = 0
    for i, job in enumerate(jobs, 1):
        print(f"RUN_EXTRACT [{i}/{len(jobs)}] stage1={job['stage1_in']} -> full={job['out_full']}", flush=True)
        try:
            n_rows = run_extract_job(
                extractor_llm=extractor_llm,
                extractor_tokenizer=extractor_tokenizer,
                extractor_sampling=extractor_sampling,
                generate_fn=run_generate,
                stage1_in=job["stage1_in"],
                out_full=job["out_full"],
                out_light=job["out_light"],
                extractor_batch_size=args.extractor_batch_size,
            )
        except Exception as e:  # one failing job must not abort the remaining ones
            n_failed += 1
            print(f"FAILED_EXTRACT stage1={job['stage1_in']} error={type(e).__name__}: {e}")
            continue
        n_ok += 1
        print(f"DONE_EXTRACT full={job['out_full']} light={job['out_light']} rows={n_rows}", flush=True)

    print(f"DONE_EXTRACT_ALL jobs={len(jobs)} ok={n_ok} failed={n_failed}")
    if n_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
