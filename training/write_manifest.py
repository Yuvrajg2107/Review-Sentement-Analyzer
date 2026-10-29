"""Hash learned artifacts and measured outputs for reproducible handoff."""
from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parents[1]
files=['models/baseline.joblib','models/transformer/model.safetensors','models/transformer/config.json','models/transformer/tokenizer.json','reports/metrics.json','reports/predictions.npz','reports/feature_weights.json']
entries=[]
for relative in files:
    path=root/relative
    if not path.exists(): continue
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    entries.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest.hexdigest()))
(root/'reports'/'artifact_manifest.json').write_text(json.dumps(dict(algorithm='SHA-256',files=entries),indent=2),encoding='utf-8')
print(f'Hashed {len(entries)} model and report artifacts')
