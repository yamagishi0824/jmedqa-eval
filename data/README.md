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

`data/*.csv` and `data/*.jsonl` are git-ignored; the dataset is not redistributed
from this repository. See the top-level README for the columns used by the
pipeline, and `src/prepare_jmedqa.py` for converting a JSONL export into the same
CSV shape.
