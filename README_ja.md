# JMedQA 評価パイプライン

日本の医師国家試験から作ったベンチマーク [JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)
（3,581 問、2018〜2026 年）で、大規模言語モデル（LLM）と視覚言語モデル（VLM）を評価するパイプラインです。

- **台帳 1 つ、ランナー 1 つ。** 評価の 1 回分が [`configs/models.tsv`](configs/models.tsv) の 1 行です。
  実行・採点・集計はすべてこの台帳を読みます。
- **モデルごとの公式設定。** システムプロンプト、thinking の切り替え、サンプリング、出力の上限は、
  各モデルの公式 model card に従います（[`src/model_profiles.py`](src/model_profiles.py)）。
- **回答形式だけを指定するプロンプト + ルールベースの採点。** プロンプトは「最後の行に `Answer: X`」と
  形式を指定するだけで、推論を促すことも禁じることもしません。解答はルールで取り出します
  （[`src/answer_parser.py`](src/answer_parser.py)）。
- **文章と画像。** 各問題を、問題文のまま（`original`）、画像への言及を除いた形（`no_image`）、
  そして VLM では試験の画像を付けた形で評価します。

[English README](README.md)

## v2 で変わったこと

[v1.0](https://github.com/yamagishi0824/jmedqa-eval/tree/v1.0) からの主な変更:

- **解答抽出の見直しと LLM 抽出の廃止。** プロンプトは回答形式（最後の行に `Answer: X`）だけを指定し、見直した
  ルールベースの抽出で最終回答を採点します。v1 の抽出 LLM の段階は廃止しました。評価したモデルでは、ルールで
  答えを取り出せた割合は 96.6〜100%（中央値 99.8%）で、回答形式の遵守率と抽出成功率もモデルごとに出します。
- **モデルごとの推論条件。** システムプロンプト、thinking の切り替え、サンプリング、出力の上限を、各モデルの
  公式 model card に従って設定します（v1 は共通のシステムプロンプトのプリセットと、全モデル greedy）。
- **VLM 対応。** 画像エンコーダを持つモデルは、試験の画像を付けて評価できます。
- **台帳と単一のランナー。** 評価の回はすべて `configs/models.tsv` で定義し、`scripts/run_eval.sh` で実行します。
- **修正。** `answer_mode` はデータセットの列を使います（v1 は選択肢の無い計算問題 25 問を多肢選択として扱い、
  不正解にしていました）。
- **モデル。** Gemma 4、Qwen3.8、DeepSeek V3.2 / V4、GLM-4.7、gpt-oss、日本語（医療）LLM などを追加しました。

v1 のコードは `v1.0` タグにあります。

## 必要なもの

- NVIDIA GPU を積んだ Linux（vLLM）。集計は CPU だけで動きます。
- Python 3.10〜3.12、[uv](https://docs.astral.sh/uv/)
- vLLM 0.30 以上、transformers 5 以上

```bash
bash scripts/setup_env.sh          # .venv を作る（既定は vLLM の +cu129 wheel。CUDA_TAG=pypi で PyPI 版）
```

実行スクリプトは、`.venv/bin/python` があればそれを、無ければ `uv run python` を使います。

ハードウェアについての注意:

- PyPI の vLLM / torch は CUDA 13 向けです。CUDA 12.x までのドライバでは `+cu129` の wheel を使ってください
  （`setup_env.sh` の既定）。
- DeepSeek V3.2 / V4 は実行時に DeepGEMM のカーネルをコンパイルするため、`nvcc` 12.9 以上が必要です（`CUDA_HOME`）。
- FP4 / MXFP4 のチェックポイント（DeepSeek-V4、`openai/gpt-oss-*`）は、Hopper 世代の GPU で CUDA 12.9 以上に
  対応したドライバ（または CUDA forward compatibility パッケージ）が必要なカーネルを使います。
- gated のリポジトリ（`meta-llama/*` など）には `huggingface-cli login` か `HF_TOKEN` が必要です。

## 環境ごとの設定

環境モジュール、キャッシュの置き場所、プロキシ、トークンなど、環境に固有の設定は、git の管理外のファイルに書きます。
どの実行スクリプトもこれを読み込みます。

```bash
cp scripts/env.example.sh scripts/env.sh   # 編集する。scripts/env.sh は git-ignore 済み
```

スクリプトは普通の `bash` で、既定では GPU 8 枚の 1 ノードを想定しています（台帳の `tp` 列。`TP_OVERRIDE` で上書き）。

## データ

データセットは Hugging Face Hub で公開しています:
[SIP-med-LLM/JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)（`jmedqa.csv` と `images/`）。

```bash
# 文章だけの評価: CSV だけで足りる
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path('data').mkdir(exist_ok=True)
shutil.copy(hf_hub_download('SIP-med-LLM/JMedQA', 'jmedqa.csv', repo_type='dataset'), 'data/jmedqa.csv')
"
# 画像ありの評価: 画像を含むデータセット全体を data/hf_jmedqa/ に
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('SIP-med-LLM/JMedQA', repo_type='dataset', local_dir='data/hf_jmedqa')
"
```

パイプラインが使う列:

| 列 | 用途 |
|---|---|
| `question_raw` | 画像への言及を残した元の問題文（`original` と画像ありの評価） |
| `question` | 画像への言及を除いた問題文（`no_image`） |
| `options_json`, `answer_json` | 選択肢（`a`〜`e`）と正答（JSON 配列） |
| `answer_mode`, `answer_count` | `option` / `numeric`、答える数 |
| `image_paths_json` | 画像の参照名 → 画像ファイル（例 `{"別冊No.1": ["images/2018/A/2018A_No01.png"]}`） |
| `image_dependency` | `none` / `enough text` / `not enough text` / `image only` / `image question` |
| `year`, `section`, `clinical_area` | 集計のキー |

各問題について `question_raw` から `original` のプロンプトを作り、`question` と異なる場合（966 問）だけ
`no_image` のプロンプトを足します。画像に関係しない問題を 2 回推論することはありません。

## 台帳: `configs/models.tsv`

1 行が 1 回の評価です（タブ区切り、空欄は `-`）。

| 列 | 意味 |
|---|---|
| `name` | 回の名前（出力ディレクトリ名の先頭）。設定違いは別の名前にする（例 `llm-jp-4-33b-thinking-effort-high`） |
| `model` | Hugging Face のリポジトリ ID、またはローカルのチェックポイント |
| `profile`, `mode` | `src/model_profiles.py` のキー、`think` / `nothink` |
| `tp` | tensor parallel の数 |
| `sampling` | `auto`（profile が推奨値ありとしていれば推奨値、無ければ greedy）、`greedy`、`official` |
| `kwargs` | `chat_template_kwargs` に上書きする JSON（例 `{"reasoning_effort":"high"}`） |
| `vision` | `1` なら画像ありでも評価する |
| `groups` | 選択用のタグ（`GROUP=<タグ>`） |
| `category`, `subgroup`, `family`, `total_B`, `active_B` | 表のための区分とパラメータ数 |

```bash
python src/registry.py list        # 全部の回の状態（done / todo）と出力先
```

新しいモデルを評価するときは、合う profile が無ければ `src/model_profiles.py` に足し、台帳に 1 行足します。

## 実行

```bash
ONLY=gemma-4-E4B-it LIMIT=10 bash scripts/run_eval.sh   # 疎通確認 -> outputs/test/
ONLY=gemma-4-E4B-it bash scripts/run_eval.sh            # 1 モデル
GROUP=qwen bash scripts/run_eval.sh                     # タグ "qwen" の行すべて
IMAGES=1 GROUP=gemma bash scripts/run_eval.sh           # タグ "gemma" の VLM を画像ありで
```

出力がすでにある回は飛ばすので、同じコマンドを投げ直せます。出力:

```text
outputs/<greedy|official>/<name>_<profile>_<mode>/
  jmedqa_stage1_full.jsonl             プロンプトごとの記録（ルールベースの予測、最終回答、トークン数）
  jmedqa_stage1_full_reasoning.jsonl   プロンプト、推論部分、出力の全文
  jmedqa_stage1_light.csv              問題ごとの正誤と診断情報
  run_config.json                      解決済みの profile と設定
outputs/vision_<greedy|official>/...   画像ありの評価（画像参照のある問題だけ）
```

### プロンプト

ユーザー発話は、回答形式の指示のあとに問題と選択肢を並べたものです。単一選択の例:

```text
次の問題に答えてください。回答の最後の行に、選んだ選択肢の記号を「Answer: X」の形式で書いてください（例: Answer: c）。

[問題]
...
[選択肢]
a. ...
```

複数選択と数値の問題では、それぞれ対応する形（`Answer: X, Y`、単位なしの数値）を指定します。
文章だけの評価では、画像参照のある問題の `original` プロンプトの先頭に「画像は利用できません」という注記を付けます。
システムプロンプトと thinking の切り替えは、モデルの profile に従います。

### 推論部分と解答の抽出

- 推論部分は、モデルごとの区切り（`</think>`、Gemma 4 のチャネル、MedGemma の `<unused95>`、Harmony の `final`
  チャネル）で切り分け、最終回答だけを採点します。
- [`src/answer_parser.py`](src/answer_parser.py) は最後の `Answer:` 行を読み、無ければ `\boxed{}`、JSON の `"answer"`、
  先頭 / 末尾の解答行、日本語・英語の最終解答の言い回し、短い答えだけの出力の順に試します。どの規則で取れたか
  （`rule_method`）を残すので、回答形式の遵守率も出せます。
- 答えを取り出せなかった問題は不正解として数えます。

### サンプリングと出力の上限

- model card がサンプリングの値を「推奨」しているモデルはその値で（`outputs/official/`）、推奨が無いモデル
  （例として値を示しているだけのものを含む）は greedy で（`outputs/greedy/`）評価します。
  サンプリングする回では、問題ごとに seed を固定します。
- 打ち切りを避けるため、出力の上限は余裕を持たせています（thinking 65,536 トークン、thinking なし 32,768。
  context の短いモデルは、問題ごとに残りの context に収めます）。打ち切りの件数も集計します。

### 画像ありの評価（`IMAGES=1`）

`vision=1` の行（vLLM が対応する画像エンコーダを持つモデル）だけを実行します。画像参照のある 944 問に、
元の問題文（`question_raw`）と画像（1 問あたり 1〜4 枚）を渡します。画像は参照名のラベル（例 `[別冊No.1Ａ]`）を
付けて問題文の前に置き、「画像は利用できません」の注記は付けません。サンプリングと出力の上限は文章の評価と同じです。

## 結果

これまでに評価したモデルの正答率です（ルールベースの採点、元の問題文）。評価したモデルが増えたら、この表を更新します。
すべての統計値は [`results/`](results/) にあります（VLM の画像依存度別の成績は
[`results/vision_by_dependency.md`](results/vision_by_dependency.md)）。

- **All**: 3,581 問、**2026**: 2026 年の 400 問。
- **With images**: VLM のみ。画像参照のある問題を画像付きで解かせた結果です（– は画像ありでは評価していない回。主に画像エンコーダの無いモデル）。
- **(think) / (no think)**: thinking を切り替えられる無印の Qwen3 は、thinking のオンとオフの両方で評価しています。
- **Sampling**: `official` = model card の推奨値、`greedy` = temperature 0（model card に推奨が無いモデル）。

#### 主な回

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
| Medical-GPT-OSS-Swallow-120B | Medical (NEDO) | official | 0.902 | 0.920 | – | – |
| GPT-OSS-Swallow-120B-SFT-v0.1 | Japanese | official | 0.901 | 0.943 | – | – |
| gemma-4-12B-it | General | official | 0.886 | 0.882 | 0.896 | 0.885 |
| gpt-oss-120b | General | official | 0.886 | 0.910 | – | – |
| Medical-Qwen3-Swallow-30B-A3B | Medical (NEDO) | official | 0.881 | 0.907 | – | – |
| Qwen3-Swallow-32B-RL-v0.2 | Japanese | official | 0.881 | 0.882 | – | – |
| Qwen3-30B-A3B-Thinking-2507 | General | official | 0.876 | 0.873 | – | – |
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
| gpt-oss-20b | General | official | 0.822 | 0.835 | – | – |
| Qwen3-Swallow-8B-RL-v0.2 | Japanese | official | 0.815 | 0.830 | – | – |
| llm-jp-4-32b-a3b-thinking | Japanese | greedy | 0.814 | 0.835 | – | – |
| GLM-4.7-Flash | General | official | 0.809 | 0.830 | – | – |
| Qwen3-32B (no think) | General | official | 0.806 | 0.835 | – | – |
| Medical-Qwen3-Swallow-32B | Medical (NEDO) | official | 0.801 | 0.868 | – | – |
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

#### reasoning effort（low / medium / high）

| Model | Effort | Sampling | All (3,581) | 2026 (400) | 推論トークン（中央値） |
|---|---|---|---|---|---|
| gpt-oss-120b | low | official | 0.852 | 0.905 | 66 |
| gpt-oss-120b | medium (default) | official | 0.886 | 0.910 | 316 |
| gpt-oss-120b | high | official | 0.904 | 0.930 | 1148 |
| llm-jp-4-33b-thinking | low | greedy | 0.797 | 0.838 | 276 |
| llm-jp-4-33b-thinking | medium (default) | greedy | 0.855 | 0.880 | 1154 |
| llm-jp-4-33b-thinking | high | greedy | 0.861 | 0.885 | 3610 |
| gpt-oss-20b | low | official | 0.735 | 0.750 | 69 |
| gpt-oss-20b | medium (default) | official | 0.822 | 0.835 | 473 |
| gpt-oss-20b | high | official | 0.848 | 0.860 | 1585 |
| llm-jp-4-32b-a3b-thinking | low | greedy | 0.763 | 0.815 | 61 |
| llm-jp-4-32b-a3b-thinking | medium (default) | greedy | 0.814 | 0.835 | 293 |
| llm-jp-4-32b-a3b-thinking | high | greedy | 0.816 | 0.828 | 977 |
| llm-jp-4-8b-thinking | low | greedy | 0.729 | 0.767 | 64 |
| llm-jp-4-8b-thinking | medium (default) | greedy | 0.784 | 0.805 | 306 |
| llm-jp-4-8b-thinking | high | greedy | 0.783 | 0.797 | 1118 |

#### 推奨値で評価したモデルの greedy の回（greedy のモデルと比べる用）

| Model | Category | Sampling | All (3,581) | 2026 (400) | With images: all | With images: 2026 |
|---|---|---|---|---|---|---|
| Qwen3-235B-A22B-Thinking-2507 | General | greedy | 0.922 | 0.930 | – | – |
| gpt-oss-120b | General | greedy | 0.892 | 0.895 | – | – |
| gemma-3-27b-it | General | greedy | 0.744 | 0.787 | – | – |

### 表の作り方

```bash
python src/summarize.py                     # -> results/（summary.md、summary.csv など）
bash scripts/run_calc_jmedqa.sh             # 詳細な表（年度、ブロック、臨床領域、変種など）
```

主な正答率は、どれも 3,581 問ベースです。

| 指標 | 定義 |
|---|---|
| `acc_original` | 全問を元の問題文で |
| `acc_no_image` | 画像参照の無い問題は original の結果、画像参照のある 966 問は `no_image` 版の結果 |
| `acc_image` | 画像ありの回: 画像参照のある 944 問は画像付きの結果、残りは original の結果 |

`summarize.py` は `summary.csv`（回ごとの 1 行。回答形式の遵守率、抽出成功率、打ち切り件数、推論・回答の
トークン数の中央値も含む）、`summary_by_year.csv`、`vision_by_dependency.csv`（画像参照のある問題を
`image_dependency` 別に、original → no_image → 画像あり）を書きます。

## ディレクトリ構成

```text
.
├── configs/models.tsv              # 評価の台帳
├── data/                           # jmedqa.csv と hf_jmedqa/（ここでは配布しない）
├── results/                        # 集計の統計値（生の出力は含めない）
├── scripts/
│   ├── env.example.sh              # 環境ごとの設定のひな形
│   ├── common.sh                   # 実行スクリプト共通の環境
│   ├── setup_env.sh                # .venv を作る
│   ├── run_eval.sh                 # 台帳の行の stage1（文章 / 画像）
│   └── run_calc_jmedqa.sh          # 詳細な表
├── src/
│   ├── model_profiles.py           # モデルごとの公式設定
│   ├── registry.py                 # configs/models.tsv を読む
│   ├── infer_jmedqa.py             # stage1（生成 + ルールベースの抽出、文章と画像）
│   ├── answer_parser.py            # ルールベースの解答抽出
│   ├── infer_jmedqa_extract_llm.py # v1 のパイプライン（LLM 抽出）。データ読み込みの関数を流用
│   ├── summarize.py                # 主な集計表
│   ├── calc_jmedqa.py              # 詳細な表
│   ├── jmedqa_variants.py          # original / no_image の展開
│   ├── prepare_jmedqa.py           # JSONL -> CSV（非公開の書き出し用）
│   └── encoding_dsv32.py           # chat template の補助（v1 のパイプライン）
└── notebooks/jmedqa_analysis.ipynb
```

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) で公開しています。

データセットは別のライセンスで、MIT License の対象ではありません。問題は厚生労働省が公開している医師国家試験の
資料に由来します。[データセットのリポジトリ](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)に記載の
ライセンス・条件と、[厚生労働省の利用規約](https://www.mhlw.go.jp/)に従ってください。
このリポジトリからデータセットは再配布しません。

## 引用

```bibtex
@misc{yamagishi2026jmedqa,
  title        = {JMedQA: Benchmarking Large Language Models and Vision-Language Models on the Japanese Medical Licensing Examination},
  author       = {Yamagishi, Yosuke and Kobayashi, Kazuma and Shibaki, Ryota and Aizawa, Akiko and Kurohashi, Sadao},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/SIP-med-LLM/JMedQA}},
  note         = {Hugging Face dataset}
}
```
