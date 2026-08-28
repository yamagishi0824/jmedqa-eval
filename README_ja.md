# JMedQA 評価パイプライン

日本の医師国家試験ベースのQAデータセット JMedQA でLLMを評価するためのリポジトリです。
次の2系統の推論に対応します。

- **正規表現抽出**: モデルに `Answer:` 形式で回答させ、正規表現で解答を取り出す
- **抽出LLM**: 自由形式で回答させ、別のLLMで最終回答を抽出する

各問題は「オリジナルの問題文」と「画像に関する記載を除去した問題文」の2通りで
推論し、同一問題のペアで画像依存性を比較できます。

[English README](README.md)

## 動作要件

- 推論にはNVIDIA GPUを備えたLinux環境（vLLM）が必要です。集計はCPUのみで動きます。
- Python 3.10〜3.12
- 依存管理に [uv](https://docs.astral.sh/uv/) を使用します

```bash
uv sync
```

モデル重みはHugging Faceから取得します。Gatedリポジトリを使う場合は
`huggingface-cli login` で認証するか、`HF_TOKEN` を環境変数に設定してください。

## 設定

実行スクリプトは特定環境に依存しない構成です。環境モジュール、キャッシュ先、
プロキシ、トークン、テンソル並列数といった環境固有の設定は、Git管理外の
ローカルファイルにまとめます。

```bash
cp scripts/env.example.sh scripts/env.sh
# scripts/env.sh を自分の環境に合わせて編集
```

`scripts/env.sh` は `.gitignore` 済みで、各実行スクリプトから自動で読み込まれます。

### Slurm

スクリプトには汎用的な `#SBATCH` 設定（1ノード8GPU）のみを記載しています。
キューや課金先はスクリプトに埋め込まず、投入時に指定してください。

```bash
sbatch --partition=<partition> --account=<account> scripts/run_eval_jmedqa_re.sh
```

単体のGPUマシンで実行する場合は `bash` で直接起動し、`TP` を利用可能なGPU数に
合わせて設定します。

## ディレクトリ構成

```text
.
├── data/
│   └── jmedqa.csv                       # 入力データ（本リポジトリでは配布しません）
├── scripts/
│   ├── env.example.sh                   # 環境固有設定のテンプレート
│   ├── run_eval_jmedqa_re.sh            # 推論 + 正規表現抽出
│   ├── run_eval_jmedqa_extract_llm.sh   # 推論 + 抽出LLM
│   └── run_calc_jmedqa.sh               # 集計
├── src/
│   ├── prepare_jmedqa.py                # JSONL → 正規化CSV
│   ├── jmedqa_variants.py               # 問題文バージョンの展開
│   ├── infer_jmedqa_re.py
│   ├── infer_jmedqa_extract_llm.py
│   ├── calc_jmedqa.py
│   └── encoding_dsv32.py                # チャットテンプレート補助
├── notebooks/
│   └── jmedqa_analysis.ipynb
└── results/                             # 生成される集計結果（Git管理外）
```

## 入力データ

データセットはHugging Face Hubで公開されています。
[SIP-med-LLM/JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)
（3,581問、対象年度2018〜2026年、split名 `benchmark`）

`jmedqa.csv` を `data/` に取得します。

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path('data').mkdir(exist_ok=True)
src = hf_hub_download('SIP-med-LLM/JMedQA', 'jmedqa.csv', repo_type='dataset')
shutil.copy(src, 'data/jmedqa.csv')
"
```

`datasets` ライブラリ経由でも同じ結果になります。

```bash
uv run python -c "
from datasets import load_dataset
load_dataset('SIP-med-LLM/JMedQA', split='benchmark').to_csv('data/jmedqa.csv')
"
```

CSVはそのまま使用でき、追加の前処理は不要です。本パイプラインが使用する列は
次のとおりです。

| 列 | 用途 |
|---|---|
| `problem_unique_id` | オリジナル版と画像なし版を対応付ける問題ID |
| `question_raw` | 画像参照を保持したオリジナルの問題文 |
| `question` | 画像に関する記載を除去した問題文 |
| `options_json` | 選択肢（`a`〜`e`） |
| `answer_json` | 正答（JSON配列） |
| `answer_mode` | `option` または `numeric` |
| `answer_count` | 想定される正答数 |
| `is_calc` | 計算問題フラグ |
| `year`、`section`、`clinical_area` | 集計軸 |
| `image_dependency` | `none` / `enough text` / `not enough text` / `image only` / `image question` |

本パイプラインはテキストのみを扱い、データセット内の `images/` は使用しません。
非公開のJSONLから同じ形式のCSVを作る場合は `src/prepare_jmedqa.py` を使います。

推論時の扱い:

- 全問題について `question_raw` を使った `original` 版を生成します。
- `question_raw` と `question` が異なる問題だけ、`question` を使った `no_image` 版も生成します。
- 通常のテキスト問題は重複させません。
- 出力には `question_variant`（`original` / `no_image`）と
  `has_no_image_variant`（対応する画像なし版の有無）が付与されます。

## 推論の実行

正規表現抽出版:

```bash
bash scripts/run_eval_jmedqa_re.sh
```

出力:

- `outputs_jmedqa_re_question_variants/{model}_{temp[_think]}/jmedqa_pred_full.jsonl`
- `outputs_jmedqa_re_question_variants/{model}_{temp[_think]}/jmedqa_pred_light.csv`

抽出LLM版:

```bash
bash scripts/run_eval_jmedqa_extract_llm.sh
```

処理は2段階です。

1. 対象モデルが自由形式で回答する
2. 抽出LLMがモデル出力だけを読み、最終回答を抽出する

抽出LLMには問題文や正答を渡しません。Light CSVには未抽出・解答数不一致などの
診断列も保存されます。

抽出段は、未処理のstage1出力をまず全て集め、1プロセスでまとめて処理します。
そのため抽出モデルのロードは評価モデルごとではなく実行ごとに1回だけです。
`jmedqa_pred_full.jsonl` と `jmedqa_pred_light.csv` が揃っているディレクトリは
対象外になり、対象が0件なら抽出モデルはロードされません。

出力:

- `outputs_jmedqa_extract_llm_question_variants/{model}_{temp[_think]}/jmedqa_pred_full.jsonl`
- `outputs_jmedqa_extract_llm_question_variants/{model}_{temp[_think]}/jmedqa_pred_light.csv`

モデル一覧、温度、テンソル並列数、最大トークン数は各推論スクリプト冒頭で設定します。
モデル欄にはHugging FaceのリポジトリIDまたはローカルのチェックポイントディレクトリを
指定できます。

## 問題文バージョンの指定

推論スクリプトの `--question-variants` には次を指定できます。

| 値 | 対象 |
|---|---|
| `both` | オリジナル全問と、存在する画像なし版（既定値） |
| `original` | オリジナル版のみ |
| `no_image` | 画像なし版が存在する問題のみ |

付属のシェルスクリプトは `both` を使用します。

## 集計

```bash
# 正規表現抽出版
bash scripts/run_calc_jmedqa.sh \
  outputs_jmedqa_re_question_variants \
  results/jmedqa_summary_re.csv

# 抽出LLM版
bash scripts/run_calc_jmedqa.sh \
  outputs_jmedqa_extract_llm_question_variants \
  results/jmedqa_summary_extract_llm.csv
```

Pythonスクリプトを直接使う場合:

```bash
uv run python src/calc_jmedqa.py \
  --input_dir outputs_jmedqa_extract_llm_question_variants \
  --output_file results/jmedqa_extract_summary.csv \
  --recursive
```

従来指標との比較可能性を保つため、`overall`、年度別、セクション別などの通常集計は
`original` のみを対象とし、画像なし版を加えても分母は変わりません。
バージョン比較用のテーブルは同一のペア集合を母集団とします。

| テーブル | 内容 |
|---|---|
| `by_question_variant` | `original` / `no_image` の精度 |
| `by_image_dependency_question_variant` | 画像依存度 × 問題文バージョン別の精度 |
| `question_variant_effect` | ペア数、両版の精度差、改善率、悪化率、不変率 |

その他の主なテーブル: `overall`、`by_year`、`by_section`、`by_clinical_area`、
`by_is_calc`、`by_answer_mode`、`violation_rate`、`extractor_effect`。

## プロンプト仕様

- 単一解答の選択問題には「最も当てはまるものを1つ」を追加します。
- 複数解答問題では解答数を追加指定せず、問題文の指示を優先します。
- オリジナル版では画像を参照できない旨の指示を付けます。
- 画像なし版は画像記載が除去済みのため、その指示を付けません。
- `is_calc` などから選択肢回答と数値回答を判定します。
- 正規表現版は最終行の `Answer:` を解析します。

## ライセンス

本リポジトリのコードは [MIT License](LICENSE) で公開しています。

データセットのライセンスはこれとは別であり、MIT License の対象外です。
試験内容の原典は厚生労働省が公開する日本の医師国家試験資料であるため、
[データセットリポジトリ](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)に
記載されたライセンス・条件と、適用される
[厚生労働省の利用条件](https://www.mhlw.go.jp/)に従ってください。
本リポジトリではデータセット自体を再配布していません。

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
