# JMedQA Evaluation Pipeline

Evaluation pipeline for large language models (LLMs) and vision-language models (VLMs) on
[JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA), a benchmark built from the Japanese National
Medical Licensing Examination (3,581 questions, exam years 2018–2026).

- **One registry, one runner.** Every evaluation run is a row in [`configs/models.tsv`](configs/models.tsv);
  running, scoring and summarizing all read it.
- **Official per-model settings.** System prompt, thinking switch, sampling and output budget follow each
  model's official card ([`src/model_profiles.py`](src/model_profiles.py)).
- **Format-only prompt + rule-based scoring.** The prompt only asks for `Answer: X` on the last line — it
  neither encourages nor suppresses reasoning — and answers are extracted by rules
  ([`src/answer_parser.py`](src/answer_parser.py)).
- **Text and vision.** Each question is evaluated as written (`original`), with image references removed
  (`no_image`), and — for VLMs — with the exam images attached.

[日本語版 README](README_ja.md)

## What's new in v2

Compared with [v1.0](https://github.com/yamagishi0824/jmedqa-eval/tree/v1.0):

- **Answer extraction revised; LLM extraction removed.** The prompt now only specifies the answer format
  (`Answer: X` on the last line), and a revised rule-based parser scores the final answer. The extractor-LLM
  stage of v1 has been removed: the rules recover answers for 96.6–100% of questions (median 99.8%) across
  the evaluated models, and format compliance and extraction rate are reported per model.
- **Per-model inference settings.** System prompt, thinking switch, sampling and output budget follow each
  model's official card, instead of shared system-prompt presets and greedy decoding for every model.
- **VLM support.** Models with an image encoder can be evaluated with the exam images attached.
- **Registry and single runner.** All runs are defined in `configs/models.tsv` and run with `scripts/run_eval.sh`.
- **Fixes.** `answer_mode` is taken from the dataset (v1 treated the 25 calculation questions without options as
  multiple choice and scored them as incorrect).
- **Models.** Gemma 4, Qwen3.8, DeepSeek V3.2 / V4, GLM-4.7, gpt-oss, llm-jp-4 / 4.1, Japanese (medical) LLMs, and more.

The v1 code is available at the `v1.0` tag.

## Requirements

- Linux with NVIDIA GPUs (vLLM). Summaries are CPU-only.
- Python 3.10–3.12, [uv](https://docs.astral.sh/uv/)
- vLLM ≥ 0.30, transformers ≥ 5

```bash
bash scripts/setup_env.sh          # creates .venv (vLLM +cu129 wheel by default; CUDA_TAG=pypi for the PyPI wheels)
```

The run scripts use `.venv/bin/python` when it exists, otherwise `uv run python`.

Hardware notes:

- The PyPI vLLM/torch wheels target CUDA 13. On drivers that only support CUDA 12.x, use the `+cu129`
  wheel (the default of `setup_env.sh`).
- DeepSeek V3.2 / V4 compile DeepGEMM kernels at run time and need `nvcc` ≥ 12.9 (`CUDA_HOME`).
- FP4 / MXFP4 checkpoints (DeepSeek-V4, `openai/gpt-oss-*`) use kernels that need a driver supporting
  CUDA ≥ 12.9 on Hopper GPUs (or the CUDA forward-compatibility package).
- Gated repositories (e.g. `meta-llama/*`) need `huggingface-cli login` or `HF_TOKEN`.

## Configuration

Site-specific settings — environment modules, cache locations, an outbound proxy, tokens — go into an
untracked file that every run script sources:

```bash
cp scripts/env.example.sh scripts/env.sh   # then edit; scripts/env.sh is git-ignored
```

The scripts are plain `bash` and assume one node with 8 GPUs by default (the `tp` column of the registry;
override with `TP_OVERRIDE`).

## Data

The dataset is published on the Hugging Face Hub:
[SIP-med-LLM/JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA) (`jmedqa.csv` + `images/`).

```bash
# text-only evaluation: the CSV is enough
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path('data').mkdir(exist_ok=True)
shutil.copy(hf_hub_download('SIP-med-LLM/JMedQA', 'jmedqa.csv', repo_type='dataset'), 'data/jmedqa.csv')
"
# vision evaluation: the full dataset including images -> data/hf_jmedqa/
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('SIP-med-LLM/JMedQA', repo_type='dataset', local_dir='data/hf_jmedqa')
"
```

Columns used by the pipeline:

| Column | Purpose |
|---|---|
| `question_raw` | Original question text, image references intact (`original`, and the vision setting) |
| `question` | Question with image references removed (`no_image`) |
| `options_json`, `answer_json` | Options (`a`–`e`) and gold answer (JSON array) |
| `answer_mode`, `answer_count` | `option` / `numeric`; expected number of answers |
| `image_paths_json` | Image reference → image file(s), e.g. `{"別冊No.1": ["images/2018/A/2018A_No01.png"]}` |
| `image_dependency` | `none` / `enough text` / `not enough text` / `image only` / `image question` |
| `year`, `section`, `clinical_area` | Aggregation keys |

For every question an `original` prompt is built from `question_raw`; a `no_image` prompt from `question` is
added only when the two differ (966 questions), so text-only questions are never inferred twice.

## The registry: `configs/models.tsv`

One row per evaluation run (tab-separated, `-` = empty):

| Column | Meaning |
|---|---|
| `name` | Run name (prefix of the output directory). Variants get their own name, e.g. `llm-jp-4-33b-thinking-effort-high` |
| `model` | Hugging Face repo id or a local checkpoint directory |
| `profile`, `mode` | Key in `src/model_profiles.py`; `think` / `nothink` |
| `tp` | Tensor-parallel size |
| `sampling` | `auto` (official values if the profile marks them as recommended, otherwise greedy), `greedy`, or `official` |
| `kwargs` | JSON merged into `chat_template_kwargs`, e.g. `{"reasoning_effort":"high"}` |
| `vision` | `1` = also evaluated with images |
| `groups` | Tags for selection (`GROUP=<tag>`) |
| `category`, `subgroup`, `family`, `total_B`, `active_B` | Labels and parameter counts for tables |

```bash
python src/registry.py list        # status (done / todo) and output directory of every run
```

To evaluate a new model, add a profile to `src/model_profiles.py` if none fits, then add a row.

## Running

```bash
ONLY=gemma-4-E4B-it LIMIT=10 bash scripts/run_eval.sh   # smoke test -> outputs/test/
ONLY=gemma-4-E4B-it bash scripts/run_eval.sh            # one model
GROUP=qwen bash scripts/run_eval.sh                     # all rows tagged "qwen"
IMAGES=1 GROUP=gemma bash scripts/run_eval.sh           # vision evaluation of the VLM rows tagged "gemma"
```

Runs whose output already exists are skipped, so the same command can be resubmitted. Outputs:

```text
outputs/<greedy|official>/<name>_<profile>_<mode>/
  jmedqa_stage1_full.jsonl             one record per prompt (rule-based prediction, final answer, token counts)
  jmedqa_stage1_full_reasoning.jsonl   prompt, reasoning part and full raw output
  jmedqa_stage1_light.csv              per-question correctness and diagnostics
  run_config.json                      resolved profile and settings
outputs/vision_<greedy|official>/...   the same for the vision setting (image-referencing questions only)
```

### Prompt

The user turn is an answer-format instruction followed by the question and options, e.g. for a
single-answer question:

```text
次の問題に答えてください。回答の最後の行に、選んだ選択肢の記号を「Answer: X」の形式で書いてください（例: Answer: c）。

[問題]
...
[選択肢]
a. ...
```

Multi-answer and numeric questions use the corresponding variants (`Answer: X, Y`, a number without unit).
In the text-only setting, `original` prompts of image-referencing questions start with a notice that the
image is not available. System prompts and thinking switches follow the model profile.

### Reasoning and answer extraction

- The reasoning part is split off by the model's delimiter (`</think>`, Gemma 4 channels, MedGemma
  `<unused95>`, Harmony `final` channel). Only the final answer is scored.
- [`src/answer_parser.py`](src/answer_parser.py) reads the last `Answer:` line, then falls back to `\boxed{}`,
  JSON `"answer"`, a leading/trailing answer line, common Japanese/English final-answer phrases, and short
  bare answers. The matched rule is recorded (`rule_method`), so format compliance is reported too.
- An answer that cannot be extracted counts as incorrect.

### Sampling and output budget

- Models whose card *recommends* sampling values are run with them (`outputs/official/`); models without
  a recommendation — including cards that only show example values — are run greedily (`outputs/greedy/`).
  For sampled runs, the seed is fixed per question.
- Output budgets are generous to avoid truncation (thinking 65,536 tokens, non-thinking 32,768, shorter for
  short-context models, where `max_tokens` is clipped per question to the remaining context).
  The number of truncated outputs is reported.

### Vision evaluation (`IMAGES=1`)

Only rows with `vision=1` (models with an image encoder supported by vLLM) are run. The image-referencing
questions (944 of 3,581) are given the original question text (`question_raw`) together with their images
(1–4 per question). Each image is preceded by its reference label (e.g. `[別冊No.1Ａ]`) and placed before the
question; the "image not available" notice is omitted. Sampling and budgets are the same as in the text runs.

## Results

Accuracy of the models evaluated so far (rule-based scoring, original question text). This table is updated as
more models are evaluated; full statistics are in [`results/`](results/) (VLMs by `image_dependency`:
[`results/vision_by_dependency.md`](results/vision_by_dependency.md)).

- **All**: the 3,581 questions; **2026**: the 400 questions of the 2026 exam.
- **With images**: VLMs only; the image-referencing questions are answered with their images (– = not evaluated with images, mostly models without an image encoder).
- **sip-jmed-llm-4-33b-dev-1005**: a development checkpoint of SIP-jmed-llm-4 (weights not public).
- **(think) / (no think)**: the original Qwen3 releases (hybrid thinking) are evaluated with thinking on and off.
- **Sampling**: `official` = the card's recommended values, `greedy` = temperature 0 (no recommendation in the card).

#### Main runs

| Model | Category | Sampling | All (3,581) | 2026 (400) | With images: all | With images: 2026 |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | General | official | 0.952 | 0.948 | – | – |
| DeepSeek-V4.1-Flash | General | official | 0.951 | 0.960 | 0.958 | 0.970 |
| Qwen3.8-Flash-Next | General | official | 0.947 | 0.945 | 0.959 | 0.953 |
| gemma-4-31B-it | General | official | 0.945 | 0.965 | 0.955 | 0.960 |
| GLM-4.7 | General | official | 0.940 | 0.948 | – | – |
| DeepSeek-V3.2 | General | official | 0.937 | 0.955 | – | – |
| gemma-4-26B-A4B-it | General | official | 0.937 | 0.950 | 0.949 | 0.950 |
| Qwen3-235B-A22B-Thinking-2507 | General | official | 0.924 | 0.930 | – | – |
| Qwen3.8-27B | General | official | 0.924 | 0.925 | 0.936 | 0.950 |
| Weblab-MedLLM-Qwen3-235B-Thinking | Medical (NEDO) | greedy | 0.924 | 0.935 | – | – |
| Weblab-MedLLM-GLM-4.7 | Medical (NEDO) | greedy | 0.924 | 0.932 | – | – |
| GPT-OSS-Swallow-120B-RL-v0.1 | Japanese | official | 0.915 | 0.953 | – | – |
| Qwen3-235B-A22B-Instruct-2507 | General | official | 0.913 | 0.920 | – | – |
| Llama-4-Maverick-17B-128E-Instruct | General | official | 0.911 | 0.915 | – | – |
| Weblab-MedLLM-gpt-oss-120b | Medical (NEDO) | greedy | 0.911 | 0.935 | – | – |
| sip-jmed-llm-4-33b-dev-1005 | Medical (SIP) | greedy | 0.907 | 0.930 | – | – |
| Medical-GPT-OSS-Swallow-120B | Medical (NEDO) | official | 0.902 | 0.920 | – | – |
| GPT-OSS-Swallow-120B-SFT-v0.1 | Japanese | official | 0.901 | 0.943 | – | – |
| gemma-4-12B-it | General | official | 0.886 | 0.882 | 0.896 | 0.885 |
| gpt-oss-120b | General | official | 0.886 | 0.910 | – | – |
| Medical-Qwen3-Swallow-30B-A3B | Medical (NEDO) | official | 0.881 | 0.907 | – | – |
| Qwen3-Swallow-32B-RL-v0.2 | Japanese | official | 0.881 | 0.882 | – | – |
| Qwen3-30B-A3B-Thinking-2507 | General | official | 0.876 | 0.873 | – | – |
| llm-jp-4.1-33b-thinking | Japanese | greedy | 0.874 | 0.915 | – | – |
| Qwen3-Swallow-30B-A3B-RL-v0.2 | Japanese | official | 0.872 | 0.905 | – | – |
| GPT-OSS-Swallow-20B-RL-v0.1 | Japanese | official | 0.867 | 0.887 | – | – |
| Qwen3-Swallow-32B-SFT-v0.2 | Japanese | official | 0.866 | 0.875 | – | – |
| Llama-4-Scout-17B-16E-Instruct | General | official | 0.860 | 0.882 | – | – |
| Qwen3-32B (think) | General | official | 0.856 | 0.880 | – | – |
| llm-jp-4-33b-thinking | Japanese | greedy | 0.855 | 0.880 | – | – |
| Qwen3-Swallow-30B-A3B-SFT-v0.2 | Japanese | official | 0.848 | 0.873 | – | – |
| GPT-OSS-Swallow-20B-SFT-v0.1 | Japanese | official | 0.846 | 0.875 | – | – |
| Qwen3-30B-A3B (think) | General | official | 0.844 | 0.843 | – | – |
| Qwen3-30B-A3B-Instruct-2507 | General | official | 0.841 | 0.873 | – | – |
| llm-jp-4.1-32b-a3b-thinking | Japanese | greedy | 0.835 | 0.882 | – | – |
| gpt-oss-20b | General | official | 0.822 | 0.835 | – | – |
| Qwen3-Swallow-8B-RL-v0.2 | Japanese | official | 0.815 | 0.830 | – | – |
| llm-jp-4-32b-a3b-thinking | Japanese | greedy | 0.814 | 0.835 | – | – |
| GLM-4.7-Flash | General | official | 0.809 | 0.830 | – | – |
| Qwen3-32B (no think) | General | official | 0.806 | 0.835 | – | – |
| Medical-Qwen3-Swallow-32B | Medical (NEDO) | official | 0.801 | 0.868 | – | – |
| llm-jp-4.1-8b-thinking | Japanese | greedy | 0.799 | 0.820 | – | – |
| medgemma-27b-it | Medical (MedGemma) | greedy | 0.793 | 0.810 | 0.792 | 0.807 |
| Medical-Qwen3-Swallow-8B | Medical (NEDO) | official | 0.789 | 0.805 | – | – |
| Qwen3-8B (think) | General | official | 0.787 | 0.812 | – | – |
| llm-jp-4-8b-thinking | Japanese | greedy | 0.784 | 0.805 | – | – |
| Llama-3.3-70B-Instruct | General | official | 0.782 | 0.805 | – | – |
| Qwen3-Swallow-8B-SFT-v0.2 | Japanese | official | 0.782 | 0.795 | – | – |
| Qwen3-30B-A3B (no think) | General | official | 0.779 | 0.797 | – | – |
| SIP-jmed-llm-3-8x13b-OP-32k-R0.1 | Medical (SIP) | greedy | 0.779 | 0.823 | – | – |
| Qwen3-4B-Thinking-2507 | General | official | 0.776 | 0.762 | – | – |
| gemma-4-E4B-it | General | official | 0.769 | 0.792 | 0.773 | 0.795 |
| SIP-jmed-llm-2-8x13b-OP-instruct | Medical (SIP) | greedy | 0.745 | 0.765 | – | – |
| gemma-3-27b-it | General | official | 0.743 | 0.790 | 0.748 | 0.790 |
| SIP-jmed-llm-3-8x13b-AC-32k-instruct | Medical (SIP) | greedy | 0.743 | 0.782 | – | – |
| SIP-jmed-llm-3-13b-OP-32k-R0.1 | Medical (SIP) | greedy | 0.700 | 0.755 | – | – |
| Qwen3-4B-Instruct-2507 | General | official | 0.688 | 0.720 | – | – |
| gemma-3-12b-it | General | official | 0.677 | 0.725 | 0.674 | 0.728 |
| Qwen3-8B (no think) | General | official | 0.666 | 0.708 | – | – |
| llm-jp-4-8b-instruct | Japanese | greedy | 0.651 | 0.695 | – | – |
| gemma-4-E2B-it | General | official | 0.634 | 0.685 | 0.639 | 0.698 |
| llm-jp-3.1-8x13b-instruct4 | Japanese | greedy | 0.595 | 0.630 | – | – |
| llm-jp-3.1-13b-instruct4 | Japanese | greedy | 0.495 | 0.530 | – | – |
| gemma-3-4b-it | General | official | 0.430 | 0.500 | 0.426 | 0.482 |

#### Reasoning effort

| Model | Effort | Sampling | All (3,581) | 2026 (400) | Median reasoning tokens |
|---|---|---|---|---|---|
| gpt-oss-120b | low | official | 0.852 | 0.905 | 66 |
| gpt-oss-120b | medium (default) | official | 0.886 | 0.910 | 316 |
| gpt-oss-120b | high | official | 0.904 | 0.930 | 1148 |
| llm-jp-4.1-33b-thinking | low | greedy | 0.839 | 0.873 | 65 |
| llm-jp-4.1-33b-thinking | medium (default) | greedy | 0.874 | 0.915 | 273 |
| llm-jp-4.1-33b-thinking | high | greedy | 0.879 | 0.897 | 963 |
| llm-jp-4-33b-thinking | low | greedy | 0.797 | 0.838 | 276 |
| llm-jp-4-33b-thinking | medium (default) | greedy | 0.855 | 0.880 | 1154 |
| llm-jp-4-33b-thinking | high | greedy | 0.861 | 0.885 | 3610 |
| gpt-oss-20b | low | official | 0.735 | 0.750 | 69 |
| gpt-oss-20b | medium (default) | official | 0.822 | 0.835 | 473 |
| gpt-oss-20b | high | official | 0.848 | 0.860 | 1585 |
| llm-jp-4.1-32b-a3b-thinking | low | greedy | 0.792 | 0.838 | 69 |
| llm-jp-4.1-32b-a3b-thinking | medium (default) | greedy | 0.835 | 0.882 | 328 |
| llm-jp-4.1-32b-a3b-thinking | high | greedy | 0.840 | 0.880 | 1186 |
| llm-jp-4-32b-a3b-thinking | low | greedy | 0.763 | 0.815 | 61 |
| llm-jp-4-32b-a3b-thinking | medium (default) | greedy | 0.814 | 0.835 | 293 |
| llm-jp-4-32b-a3b-thinking | high | greedy | 0.816 | 0.828 | 977 |
| llm-jp-4.1-8b-thinking | low | greedy | 0.761 | 0.807 | 68 |
| llm-jp-4.1-8b-thinking | medium (default) | greedy | 0.799 | 0.820 | 289 |
| llm-jp-4.1-8b-thinking | high | greedy | 0.811 | 0.830 | 1128 |
| llm-jp-4-8b-thinking | low | greedy | 0.729 | 0.767 | 64 |
| llm-jp-4-8b-thinking | medium (default) | greedy | 0.784 | 0.805 | 306 |
| llm-jp-4-8b-thinking | high | greedy | 0.783 | 0.797 | 1118 |

#### Greedy runs of sampled models (for comparisons with greedy-only models)

| Model | Category | Sampling | All (3,581) | 2026 (400) | With images: all | With images: 2026 |
|---|---|---|---|---|---|---|
| Qwen3-235B-A22B-Thinking-2507 | General | greedy | 0.922 | 0.930 | – | – |
| gpt-oss-120b | General | greedy | 0.892 | 0.895 | – | – |
| gemma-3-27b-it | General | greedy | 0.744 | 0.787 | – | – |

### Reproducing the tables

```bash
python src/summarize.py                     # -> results/ (summary.md, summary.csv, ...)
bash scripts/run_calc_jmedqa.sh             # detailed tables (year, section, clinical area, variants, ...)
```

All headline numbers are on the 3,581 source questions:

| Metric | Definition |
|---|---|
| `acc_original` | Original question text for every question |
| `acc_no_image` | Questions without image references: original result; the 966 image-referencing questions: their `no_image` variant |
| `acc_image` | Vision runs: the 944 image-referencing questions answered with images; the rest: original result |

`summarize.py` writes `summary.csv` (one row per run, with format compliance, extraction rate, truncation and
median reasoning / answer tokens), `summary_by_year.csv`, and `vision_by_dependency.csv` (image-referencing
questions by `image_dependency`: original → no_image → with images).

## Repository layout

```text
.
├── configs/models.tsv              # evaluation registry
├── data/                           # jmedqa.csv and hf_jmedqa/ (not distributed here)
├── results/                        # summary statistics (no raw outputs)
├── scripts/
│   ├── env.example.sh              # template for site-specific settings
│   ├── common.sh                   # shared environment for the run scripts
│   ├── setup_env.sh                # creates .venv
│   ├── run_eval.sh                 # stage1 for registry rows (text / vision)
│   └── run_calc_jmedqa.sh          # detailed tables
├── src/
│   ├── model_profiles.py           # official per-model settings
│   ├── registry.py                 # reads configs/models.tsv
│   ├── infer_jmedqa.py             # stage1 (generation + rule-based extraction, text and images)
│   ├── answer_parser.py            # rule-based answer extraction
│   ├── infer_jmedqa_extract_llm.py # v1 pipeline (LLM extraction); its data helpers are reused
│   ├── summarize.py                # headline tables
│   ├── calc_jmedqa.py              # detailed tables
│   ├── jmedqa_variants.py          # original / no_image expansion
│   ├── prepare_jmedqa.py           # JSONL -> CSV (for private exports)
│   └── encoding_dsv32.py           # chat-template helper (v1 pipeline)
└── notebooks/jmedqa_analysis.ipynb
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The dataset is licensed separately and is **not** covered by the MIT License. The exam content originates from
Japanese National Medical Examination materials published by Japan's Ministry of Health, Labour and Welfare
(MHLW); follow the license and terms stated on the
[dataset repository](https://huggingface.co/datasets/SIP-med-LLM/JMedQA) and the applicable
[MHLW terms of use](https://www.mhlw.go.jp/). The dataset is not redistributed from this repository.

## Citation

```bibtex
@misc{yamagishi2026jmedqa,
  title        = {JMedQA: Benchmarking Large Language Models and Vision-Language Models on the Japanese Medical Licensing Examination},
  author       = {Yamagishi, Yosuke and Kobayashi, Kazuma and Shibaki, Ryota and Aizawa, Akiko and Kurohashi, Sadao},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/SIP-med-LLM/JMedQA}},
  note         = {Hugging Face dataset}
}
```
