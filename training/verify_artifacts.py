"""Check that saved results agree with independent reconstruction from predictions."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'reports'/'metrics.json').read_text())
arrays=np.load(ROOT/'reports'/'predictions.npz')
y=arrays['labels']
assert len(y)==report['dataset']['test_count']==25000
assert np.bincount(y).tolist()==[12500,12500]
for model in report['models']:
    probabilities=arrays[model['id']+'_probabilities']
    assert probabilities.shape==y.shape and np.isfinite(probabilities).all()
    assert ((probabilities>=0)&(probabilities<=1)).all()
    predicted=(probabilities>=.5).astype(int)
    assert confusion_matrix(y,predicted,labels=[0,1]).tolist()==model['confusion_matrix']
    assert abs(accuracy_score(y,predicted)-model['accuracy'])<1e-12
    p,r,f,_=precision_recall_fscore_support(y,predicted,average='binary')
    assert all(abs(a-b)<1e-12 for a,b in zip([p,r,f],[model['precision'],model['recall'],model['f1']]))
    print(model['id'], 'saved metrics independently verified; test rows=',len(y))
assert len(report['models'])==2, 'Both model results must be present'
print('PASS: shared test labels, probability bounds, confusion matrices and all requested metrics')
