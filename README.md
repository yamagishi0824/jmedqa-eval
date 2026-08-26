# JMedQA Evaluation Pipeline

Evaluation pipeline for LLMs on JMedQA, a Japanese medical licensing exam QA set.
Two inference pipelines are supported:

- **Regex extraction** — the model is instructed to end its reply with `Answer:`
- **LLM extraction** — the model answers freely and a second model extracts the final answer

Each question is evaluated in two forms: the original wording and a variant with
image references removed, so image dependence can be measured on a paired cohort.

[日本語版 README](README_ja.md)

## Requirements

- Linux with NVIDIA GPUs for inference (vLLM). The aggregation step is CPU-only.
- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

```bash
uv sync
```

Model weights are pulled from Hugging Face. For gated repositories, authenticate
first (`huggingface-cli login`) or export `HF_TOKEN` in your shell.

## Configuration

The run scripts are environment-agnostic. Anything site-specific — environment
modules, cache locations, an outbound proxy, tokens, tensor-parallel size — goes
into an untracked local file:

```bash
cp scripts/env.example.sh scripts/env.sh
# edit scripts/env.sh
```

`scripts/env.sh` is git-ignored and sourced automatically by every run script.

### Slurm

The scripts carry generic `#SBATCH` directives (8 GPUs on one node). Queue and
billing options are intentionally not baked in; pass them at submit time:

```bash
sbatch --partition=<partition> --account=<account> scripts/run_eval_jmedqa_re.sh
```

On a standalone GPU machine, run the same script with `bash` and set `TP` to the
number of available GPUs.

## Repository layout

```text
.
├── data/
│   └── jmedqa.csv                       # input dataset (not distributed here)
├── scripts/
│   ├── env.example.sh                   # template for site-specific settings
│   ├── run_eval_jmedqa_re.sh            # inference + regex extraction
│   ├── run_eval_jmedqa_extract_llm.sh   # inference + extractor LLM
│   └── run_calc_jmedqa.sh               # aggregation
├── src/
│   ├── prepare_jmedqa.py                # JSONL -> normalized CSV
│   ├── jmedqa_variants.py               # question-variant expansion
│   ├── infer_jmedqa_re.py
│   ├── infer_jmedqa_extract_llm.py
│   ├── calc_jmedqa.py
│   └── encoding_dsv32.py                # chat-template helper
├── notebooks/
│   └── jmedqa_analysis.ipynb
└── results/                             # generated summaries (git-ignored)
```

## Input data

The dataset is published on the Hugging Face Hub:
[SIP-med-LLM/JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)
(3,581 questions, exam years 2018–2026, split name `benchmark`).

Download `jmedqa.csv` into `data/`:

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path('data').mkdir(exist_ok=True)
src = hf_hub_download('SIP-med-LLM/JMedQA', 'jmedqa.csv', repo_type='dataset')
shutil.copy(src, 'data/jmedqa.csv')
"
```

Equivalently, via the `datasets` library:

```bash
uv run python -c "
from datasets import load_dataset
load_dataset('SIP-med-LLM/JMedQA', split='benchmark').to_csv('data/jmedqa.csv')
"
```

The CSV is used as-is; no further preprocessing is required. Columns used by this
pipeline:

| Column | Purpose |
|---|---|
| `problem_unique_id` | Joins the original and no-image versions |
| `question_raw` | Original exam question, image references intact |
| `question` | Question with image references removed |
| `options_json` | Answer options (`a`–`e`) |
| `answer_json` | Gold answer (JSON array) |
| `answer_mode` | `option` or `numeric` |
| `answer_count` | Expected number of answers |
| `is_calc` | Calculation-question flag |
| `year`, `section`, `clinical_area` | Aggregation keys |
| `image_dependency` | `none` / `enough text` / `not enough text` / `image only` / `image question` |

This pipeline is text-only and does not consume the `images/` directory of the
dataset. `src/prepare_jmedqa.py` remains available for converting a private JSONL
export into the same CSV shape.

For every source row, inference builds an `original` prompt from `question_raw`.
A `no_image` prompt from `question` is added only when the two fields differ, so
text-only questions are never duplicated. Output records carry:

- `question_variant`: `original` or `no_image`
- `has_no_image_variant`: whether a matching no-image version exists

## Running inference

Regex pipeline:

```bash
bash scripts/run_eval_jmedqa_re.sh
```

Outputs:

- `outputs_jmedqa_re_question_variants/{model}_{temp[_think]}/jmedqa_pred_full.jsonl`
- `outputs_jmedqa_re_question_variants/{model}_{temp[_think]}/jmedqa_pred_light.csv`

Extractor-LLM pipeline:

```bash
bash scripts/run_eval_jmedqa_extract_llm.sh
```

Two stages: the evaluated model produces a free-form answer, then the extractor
LLM reads *only* that output and extracts the final answer. The question and the
gold answer are never passed to the extractor. The light CSV also records
extraction diagnostics such as parse failures and answer-count mismatches.

Outputs:

- `outputs_jmedqa_extract_llm_question_variants/{model}_{temp[_think]}/jmedqa_pred_full.jsonl`
- `outputs_jmedqa_extract_llm_question_variants/{model}_{temp[_think]}/jmedqa_pred_light.csv`

Models, temperatures, tensor parallelism, and token limits are configured at the
top of each inference script. Model entries accept a Hugging Face repo ID or a
local checkpoint directory.

## Selecting question variants

Both inference scripts accept `--question-variants`:

| Value | Rows inferred |
|---|---|
| `both` | All originals plus available no-image variants (default) |
| `original` | Original questions only |
| `no_image` | Questions having a no-image variant only |

The provided shell scripts use `both`.

## Aggregation

```bash
# Regex pipeline
bash scripts/run_calc_jmedqa.sh \
  outputs_jmedqa_re_question_variants \
  results/jmedqa_summary_re.csv

# Extractor-LLM pipeline
bash scripts/run_calc_jmedqa.sh \
  outputs_jmedqa_extract_llm_question_variants \
  results/jmedqa_summary_extract_llm.csv
```

Direct invocation:

```bash
uv run python src/calc_jmedqa.py \
  --input_dir outputs_jmedqa_extract_llm_question_variants \
  --output_file results/jmedqa_extract_summary.csv \
  --recursive
```

Standard tables (`overall`, yearly, section accuracy) use `original` rows only,
so adding no-image rows does not change their denominators. Variant-specific
tables use the paired cohort:

| Table | Contents |
|---|---|
| `by_question_variant` | Accuracy for `original` and `no_image` |
| `by_image_dependency_question_variant` | Accuracy by image dependency and variant |
| `question_variant_effect` | Pair count, accuracy delta, improvement, degradation, unchanged rates |

Other tables: `overall`, `by_year`, `by_section`, `by_clinical_area`,
`by_is_calc`, `by_answer_mode`, `violation_rate`, `extractor_effect`.

## Prompt behavior

- Single-answer option questions receive a "choose the single best answer" instruction.
- Multi-answer questions rely on the count stated in the question.
- Original image-referenced questions keep the text-only instruction.
- No-image variants omit it, since image references are already removed.
- Option and numeric answer modes are handled separately.

## License

The code in this repository is released under the [MIT License](LICENSE).

The dataset is licensed separately and is **not** covered by the MIT License.
The exam content originates from Japanese National Medical Examination materials
published by Japan's Ministry of Health, Labour and Welfare (MHLW); follow the
license and terms stated on the
[dataset repository](https://huggingface.co/datasets/SIP-med-LLM/JMedQA) and the
applicable [MHLW terms of use](https://www.mhlw.go.jp/). The dataset is not
redistributed from this repository.

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
