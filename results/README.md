# results/

Summary statistics of the runs in `configs/models.tsv` (no raw model outputs). Regenerated with
`python src/summarize.py` and updated as more models are evaluated.

| File | Contents |
|---|---|
| `summary.md` | The tables shown in the top-level README |
| `summary.csv` | One row per run: accuracy (original / no_image / with images; all questions and 2026), format compliance (`answer_line_rate`), extraction rate, truncated outputs, median reasoning / answer tokens, parameter counts |
| `summary_by_year.csv` | Accuracy per exam year (original, and with images for VLMs) |
| `vision_by_dependency.csv` | VLMs: image-referencing questions by `image_dependency` (original → no_image → with images, and the gain with images in points) |
| `vision_by_dependency.md` | The same as a Markdown table (original → with images, gain in points) |

All accuracies are on the 3,581 source questions (2026: 400 questions); see the top-level README for the definitions.
Scoring is rule-based (`src/answer_parser.py`); an unextractable answer counts as incorrect.
