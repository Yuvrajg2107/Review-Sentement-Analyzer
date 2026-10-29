"""Export fitted coefficient evidence without retraining or reading test labels."""
from pathlib import Path
import json, joblib, numpy as np
root=Path(__file__).resolve().parents[1]
estimator=joblib.load(root/'models'/'baseline.joblib')
names=estimator.named_steps['tfidf'].get_feature_names_out()
coefs=estimator.named_steps['classifier'].coef_[0]
order=np.argsort(coefs)
result=dict(negative=[dict(feature=str(names[i]),coefficient=float(coefs[i])) for i in order[:20]],positive=[dict(feature=str(names[i]),coefficient=float(coefs[i])) for i in order[-20:][::-1]],intercept=estimator.named_steps['classifier'].intercept_.tolist(),classes=estimator.classes_.tolist(),total_features=len(names),note='Global fitted coefficients, not causal effects or per-review explanations. Feature vector values also affect each prediction.')
(root/'reports'/'feature_weights.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(f'Exported {len(names)} learned feature coefficients; top20 positive/negative recorded')
