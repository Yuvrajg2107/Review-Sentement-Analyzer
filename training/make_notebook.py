"""Create a portable notebook entry point without embedding downloaded reviews."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
def markdown(source): return dict(cell_type='markdown',metadata={},source=source.splitlines(True))
def code(source): return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))
cells=[
    markdown('# Review Sentiment Analyzer\nTrain-only model selection, then one matched IMDb test comparison. This notebook calls the same versioned script as the command line.'),
    code('from pathlib import Path\nimport os\nroot = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\nos.chdir(root)\nprint(root)'),
    code('%run training/run_experiment.py --fresh'),
    code('import json, pandas as pd\nreport = json.loads(Path("reports/metrics.json").read_text())\npd.DataFrame(report["models"])[["name", "accuracy", "precision", "recall", "f1"]]'),
    code('from IPython.display import Image, display\ndisplay(Image("reports/baseline_confusion_matrix.png"))\ndisplay(Image("reports/transformer_confusion_matrix.png"))'),
    markdown('## Interpretation\nPositive=1; negative=0. Matrix rows are true labels and columns are predicted labels. Confidence is an uncalibrated probability. The pinned upstream notebook trains on the official IMDb train split and evaluates on the official test split for one epoch. It does not pin a historical dataset revision, so byte-identical data cannot be guaranteed. Read reports/training.log and data/provenance.json for actual run evidence and pinned revisions. --fresh archives saved results before retraining; omit it to resume.')
]
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=5)
(root/'notebooks'/'experiment.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
