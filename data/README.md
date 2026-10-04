# data/

Place the JMedQA dataset here as `data/jmedqa.csv`.

Download it from the Hugging Face Hub
([SIP-med-LLM/JMedQA](https://huggingface.co/datasets/SIP-med-LLM/JMedQA)):

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path('data').mkdir(exist_ok=True)
src = hf_hub_download('SIP-med-LLM/JMedQA', 'jmedqa.csv', repo_type='dataset')
shutil.copy(src, 'data/jmedqa.csv')
"
```

For the vision evaluation, download the full dataset including `images/` into `data/hf_jmedqa/`
(or point `JMEDQA_IMAGE_ROOT` at an existing copy):

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('SIP-med-LLM/JMedQA', repo_type='dataset', local_dir='data/hf_jmedqa')
"
```

`data/*.csv`, `data/*.jsonl` and `data/hf_jmedqa/` are git-ignored; the dataset is not redistributed
from this repository. See the top-level README for the columns used by the
pipeline, and `src/prepare_jmedqa.py` for converting a JSONL export into the same
CSV shape.
