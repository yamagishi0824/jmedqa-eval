#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-model inference settings ("profiles").

Every profile follows the official model card / repository of the model.
Profiles with different settings for thinking on/off are split into mode="think" / "nothink".

Keys
- system_prompt         : None means no system message (no official recommendation).
- chat_template_kwargs  : passed to the chat template (e.g. thinking switch, reasoning effort).
- sampling              : the officially recommended sampling (vLLM SamplingParams kwargs), used with --sampling official.
- official_recommended  : whether the official card actually *recommends* these values. Models whose card only shows
                          example values (or nothing) are marked False and are evaluated greedily (temperature 0).
- max_tokens / max_model_len : generous budgets to avoid truncation
                          (thinking 65536 / non-thinking 32768, shorter where the context window is shorter).
- clip_max_tokens_to_context : for short-context models, max_tokens is set per question to (context - prompt length).
- llm_kwargs            : extra kwargs for vllm.LLM(...).
- prompt_style          : how the user turn is built (see infer_jmedqa.build_user_text). All profiles use answer_line:
    answer_line         an answer-format instruction only ("write `Answer: X` on the last line") + question + options.
                        It neither encourages nor suppresses reasoning.
    legacy_ja           the free-form Japanese instruction of the v1 pipeline.
    qwen_answer         question + options + Qwen's official multiple-choice format ("answer" field).
    medgemma_final      question + options + the MedQA instruction of the MedGemma 1.5 report ('Final Answer: X').
- reasoning             : delimiter of the reasoning part (see infer_jmedqa.split_reasoning).
- redecode              : rebuild the output with tokenizer.decode(token_ids) instead of output.text (llm-jp-4 cookbook).
- sampling_extra        : always added to the sampling params (e.g. stop_token_ids).
"""

from typing import Any, Dict

_GEMMA4_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
_QWEN_THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
_QWEN_NOTHINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
# Text-only by default; image limits are added at run time for the vision evaluation (--with-images).
_TEXT_ONLY: Dict[str, Any] = {}

_SIP_SYSTEM = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"


PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ── Gemma 4 (google/gemma-4-*-it) ────────────────────────────────────────
    # Card: temperature=1.0, top_p=0.95, top_k=64 for all use cases. Thinking is switched by the template
    # (enable_thinking, default off). No recommended system prompt.
    "gemma4": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": _GEMMA4_SAMPLING,
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "gemma4",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "sampling": _GEMMA4_SAMPLING,
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "gemma4",  # 26B/31B emit an empty thought block even when off
        },
    },
    # ── MedGemma (google/medgemma-*) ─────────────────────────────────────────
    # Card example: system "You are a helpful medical assistant.", greedy decoding. MedGemma 1.5 report: temperature 0,
    # 27B thinks with "SYSTEM INSTRUCTION: think silently if needed." (delimiters <unused94>...<unused95>).
    "medgemma": {
        "nothink": {
            "system_prompt": "You are a helpful medical assistant.",
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.0},
            "max_tokens": 16384, "max_model_len": 20480,
            "llm_kwargs": _TEXT_ONLY,
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "medgemma",
        },
        "think": {
            "system_prompt": "SYSTEM INSTRUCTION: think silently if needed. You are a helpful medical assistant.",
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.0},
            "max_tokens": 16384, "max_model_len": 20480,
            "llm_kwargs": _TEXT_ONLY,
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "medgemma",
        },
    },
    # ── Qwen3.5 (0.8B-122B) ──────────────────────────────────────────────────
    # Card: thinking T=1.0/top_p=0.95/top_k=20/min_p=0/presence_penalty=1.5, non-thinking T=0.7/top_p=0.8/top_k=20/pp=1.5.
    "qwen35": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {**_QWEN_THINK, "presence_penalty": 1.5},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "sampling": _QWEN_NOTHINK,
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # Qwen3.5-397B-A17B: thinking T=0.6/top_p=0.95/top_k=20.
    "qwen35_397b": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── Qwen3.6 / Qwen3.8 / Qwen3.8-Flash-Next ───────────────────────────────
    # Card: thinking T=1.0/top_p=0.95/top_k=20/pp=0.0, non-thinking T=0.7/top_p=0.8/top_k=20/pp=1.5.
    # Qwen3.8 also has reasoning_effort (default xhigh); the default is kept.
    "qwen36": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": _QWEN_THINK,
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "sampling": _QWEN_NOTHINK,
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": _TEXT_ONLY,
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── DeepSeek ─────────────────────────────────────────────────────────────
    # V3.2: no Jinja template -> vLLM tokenizer_mode=deepseek_v32. Thinking via chat_template_kwargs. T=1.0/top_p=0.95.
    "deepseek_v32": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": True},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {"tokenizer_mode": "deepseek_v32"},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": False},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {"tokenizer_mode": "deepseek_v32"},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # V3.1: no recommendation in the card -> generation_config.json (T=0.6/top_p=0.95).
    "deepseek_v31": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": True},
            "sampling": {"temperature": 0.6, "top_p": 0.95},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": False},
            "sampling": {"temperature": 0.6, "top_p": 0.95},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # V4-Flash / V4-Flash-0731 (284B, 13B active, FP4+FP8): no Jinja template -> tokenizer_mode=deepseek_v4.
    # Card: T=1.0, top_p=1.0; vLLM recipe: --kv-cache-dtype fp8 --block-size 256.
    # Note: the FP4 expert kernels need a driver that supports CUDA >= 12.9 (see README).
    "deepseek_v4": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
            "sampling": {"temperature": 1.0, "top_p": 1.0},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {"tokenizer_mode": "deepseek_v4", "kv_cache_dtype": "fp8", "block_size": 256},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": False},
            "sampling": {"temperature": 1.0, "top_p": 1.0},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {"tokenizer_mode": "deepseek_v4", "kv_cache_dtype": "fp8", "block_size": 256},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # V4.1-Flash (552B, multimodal): tokenizer_mode=deepseek_v41. reasoning_effort low/high(default)/xhigh/max.
    # DeepSeek's benchmark setting is T=1.0, top_p=0.95; the default effort (high) is kept.
    "deepseek_v41": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {"tokenizer_mode": "deepseek_v41"},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"thinking": False},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {"tokenizer_mode": "deepseek_v41"},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── Qwen3 2507 ───────────────────────────────────────────────────────────
    # *-Thinking-2507: T=0.6, top_p=0.95, top_k=20, min_p=0; the template always opens <think>.
    "qwen3_2507": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        # *-Instruct-2507 (no thinking): T=0.7, TopP=0.8, TopK=20, MinP=0.
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── Qwen3 (2504, hybrid thinking: Qwen3-8B / 30B-A3B / 32B) ──────────────
    # Card / generation_config: thinking T=0.6, top_p=0.95, top_k=20, min_p=0; non-thinking T=0.7, top_p=0.8.
    "qwen3_hybrid": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 32768, "max_model_len": 40960,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
            "max_tokens": 32768, "max_model_len": 40960,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── llm-jp-4 (8b / 32b-a3b / 33b) ────────────────────────────────────────
    # Cookbook (github.com/llm-jp/llm-jp-4-cookbook): T=0.7, top_p=0.9 are example values (-> greedy here).
    # reasoning_effort low / medium (default) / high via the chat template; trust_remote_code is required;
    # Harmony-style output, decoded with tokenizer.decode(token_ids).
    "llmjp4": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "sampling": {"temperature": 0.7, "top_p": 0.9},
            "max_tokens": 61440, "max_model_len": 65536,
            "llm_kwargs": {"trust_remote_code": True},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "harmony", "redecode": True,
        },
        # llm-jp-4-*-instruct (no thinking, Harmony-compatible template)
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.7, "top_p": 0.9},
            "max_tokens": 61440, "max_model_len": 65536,
            "llm_kwargs": {"trust_remote_code": True},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "harmony", "redecode": True,
        },
    },
    # ── SIP-jmed-llm (SIP-med-LLM/*) and llm-jp-3.1 ──────────────────────────
    # The chat template / card example use a fixed Japanese system prompt; card sampling values are examples (-> greedy).
    "sipjmed3": {   # SIP-jmed-llm-3 (13B / 8x13B, 32k context)
        "nothink": {
            "system_prompt": _SIP_SYSTEM,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05},
            "max_tokens": 28672, "max_model_len": 32768,
            "llm_kwargs": {},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    "sipjmed2": {   # SIP-jmed-llm-2 (4k context)
        "nothink": {
            "system_prompt": _SIP_SYSTEM,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05},
            "max_tokens": 4096, "max_model_len": 4096, "clip_max_tokens_to_context": True,
            "llm_kwargs": {},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    "llmjp31": {    # llm-jp-3.1-*-instruct4 (4k context)
        "nothink": {
            "system_prompt": _SIP_SYSTEM,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05},
            "max_tokens": 4096, "max_model_len": 4096, "clip_max_tokens_to_context": True,
            "llm_kwargs": {},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── GLM ──────────────────────────────────────────────────────────────────
    # GLM-4.7-Flash: card evaluation setting temperature=1.0, top_p=0.95. Thinking on by default.
    "glm47": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # GLM-4.7 (355B MoE): same evaluation setting; expert parallelism.
    "glm47_full": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 61440, "max_model_len": 65536,
            "llm_kwargs": {"enable_expert_parallel": True},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── Japanese medical LLMs (tokyotech-llm Medical-*-Swallow, weblab-LLM-M Weblab-MedLLM) ──
    # Medical-Qwen3-Swallow / Qwen3-Swallow: the card recommends generation_config.json (T=0.6, top_p=0.95, top_k=20, min_p=0)
    # and a maximum context of 32,768 tokens. Thinking on (Qwen3 template).
    "medswallow_qwen3": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 28672, "max_model_len": 32768,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # (Medical-)GPT-OSS-Swallow (BF16): same generation_config recommendation; Harmony; reasoning_effort medium (default).
    "medswallow_gptoss": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 28672, "max_model_len": 32768,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "harmony",
        },
    },
    # Weblab-MedLLM-*: no sampling recommendation in the cards (examples only) -> greedy. No system prompt.
    "weblab_qwen3_thinking": {   # based on Qwen3-235B-A22B-Thinking-2507; the template always opens <think>
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    "weblab_qwen3_instruct": {   # gated; non-thinking
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    "weblab_gptoss": {   # based on gpt-oss-120b; F32 weights -> dtype=bfloat16 as in the card
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "sampling": {"temperature": 1.0, "top_p": 1.0},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {"dtype": "bfloat16"},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "harmony",
        },
    },
    "weblab_glm47": {   # based on GLM-4.7; the card recommends max-model-len 65536 and expert parallelism
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "max_tokens": 61440, "max_model_len": 65536,
            "llm_kwargs": {"enable_expert_parallel": True},
            "official_recommended": False, "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # ── Others ───────────────────────────────────────────────────────────────
    # openai/gpt-oss-*: card "We recommend sampling with temperature=1.0 and top_p=1.0". Harmony; reasoning_effort medium (default).
    # Note: the MXFP4 kernels need a driver that supports CUDA >= 12.9 on Hopper (see README).
    "gptoss_openai": {
        "think": {
            "system_prompt": None,
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "sampling": {"temperature": 1.0, "top_p": 1.0},
            "max_tokens": 65536, "max_model_len": 69632,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "harmony",
        },
    },
    # Gemma 3 (4B / 12B / 27B): no thinking. generation_config: top_k=64, top_p=0.95 (temperature 1.0).
    "gemma3": {
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
    # Llama 3.3 / 4: no recommendation in the card; the distributed generation_config (T=0.6, top_p=0.9) is used.
    "llama_gc": {
        "nothink": {
            "system_prompt": None,
            "chat_template_kwargs": {},
            "sampling": {"temperature": 0.6, "top_p": 0.9},
            "max_tokens": 32768, "max_model_len": 36864,
            "llm_kwargs": {},
            "prompt_style": "answer_line", "reasoning": "think_tag",
        },
    },
}


GREEDY = {"temperature": 0.0}


def get_profile(name: str, mode: str, sampling: str = "greedy") -> Dict[str, Any]:
    """Return a copy of a profile. sampling="greedy" (temperature 0) or "official" (the card's values)."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile: {name} (choices: {', '.join(PROFILES)})")
    if mode not in PROFILES[name]:
        raise KeyError(f"profile {name} has no mode {mode}")
    if sampling not in ("greedy", "official"):
        raise ValueError(f"unknown sampling: {sampling}")
    prof = dict(PROFILES[name][mode])
    prof["sampling_official"] = dict(prof["sampling"])
    if sampling == "greedy":
        prof["sampling"] = dict(GREEDY)
    # Settings added regardless of the sampling mode (e.g. stop_token_ids)
    prof["sampling"].update(prof.get("sampling_extra", {}))
    prof["sampling_mode"] = sampling
    prof.setdefault("official_recommended", True)
    return prof
