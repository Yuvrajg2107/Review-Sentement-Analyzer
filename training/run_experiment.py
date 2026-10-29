"""Reproducible IMDb comparison. Test labels are used only after model selection."""
from pathlib import Path
import os
import sys
if sys.platform == 'win32':
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass  # Standard CA verification remains enabled when truststore is absent.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tmp' / 'matplotlib'))
import argparse, json, time, hashlib, platform, gc, shutil
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss

SEED = 42
DATASET = 'stanfordnlp/imdb'
MODEL = 'lvwerra/distilbert-imdb'
DATASET_REVISION = 'e6281661ce1c48d982bc483cf8a173c1bbeb5d31'
MODEL_REVISION = '0fc02cd68445b599a9cb2da2368050e7fb31d29a'

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def log(message):
    line = f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}'
    print(line, flush=True)
    with (ROOT / 'reports' / 'training.log').open('a', encoding='utf-8') as file:
        file.write(line + '\n')

def pipeline(kind, c):
    word = TfidfVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,2),
                           min_df=2, max_df=.98, max_features=100000, sublinear_tf=True,
                           dtype=np.float32)
    features = word if kind == 'word' else FeatureUnion([
        ('word', word),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3,
                               max_features=60000, sublinear_tf=True, dtype=np.float32))
    ], transformer_weights={'word':1.0,'char':0.5})
    return Pipeline([('tfidf', features), ('classifier', LogisticRegression(
        C=c, max_iter=500, solver='liblinear', random_state=SEED))])

def scores(y, probabilities):
    pred = (np.asarray(probabilities) >= .5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return dict(accuracy=float(accuracy_score(y,pred)), precision=float(precision),
                recall=float(recall), f1=float(f1), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                log_loss=float(log_loss(y,np.column_stack([1-np.asarray(probabilities),probabilities]))))

def plot_reports(metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#4E342E','axes.labelcolor':'#4E342E'})
    cmap=LinearSegmentedColormap.from_list('cocoa',['#F8F4E7','#CC5500','#4E342E'])
    for model in metrics['models']:
        fig, ax=plt.subplots(figsize=(6,5),facecolor='#F8F4E7')
        cm=np.array(model['confusion_matrix']); ax.imshow(cm,cmap=cmap)
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=22,color='white' if cm[i,j]>cm.max()/2 else '#4E342E')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Predicted label',ylabel='Actual label',title=model['name']+'\nIMDb official test split')
        fig.tight_layout(); fig.savefig(ROOT/'reports'/f"{model['id']}_confusion_matrix.png",dpi=160); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--skip-transformer',action='store_true')
    parser.add_argument('--fresh',action='store_true',help='Archive previous results and baseline, then retrain from scratch')
    parser.add_argument('--dataset-revision',default=DATASET_REVISION)
    parser.add_argument('--model-revision',default=MODEL_REVISION)
    args=parser.parse_args()
    for folder in ['reports','data','models','notebooks']: (ROOT/folder).mkdir(exist_ok=True)
    if args.fresh:
        archive=ROOT/'reports'/'archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        for relative in ['reports/selection.json','reports/validation_progress.json','reports/run_signature.json','reports/baseline_probabilities.npy','reports/transformer_probabilities.npy','reports/metrics.json','reports/predictions.npz','reports/training.log','models/baseline.joblib']:
            path=ROOT/relative
            if path.exists(): shutil.move(str(path),str(archive/path.name))
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','joblib','transformers','datasets','huggingface-hub','torch','matplotlib']}
    write_json(ROOT/'reports'/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions))
    api=HfApi(); dataset_revision=api.dataset_info(DATASET,revision=args.dataset_revision).sha; model_revision=api.model_info(MODEL,revision=args.model_revision).sha
    log(f'Dataset {DATASET}@{dataset_revision}; transformer {MODEL}@{model_revision}')
    dataset=load_dataset(DATASET,revision=dataset_revision)
    train=dataset['train']; test=dataset['test']; x=np.array(train['text'],dtype=object); y=np.array(train['label']); xt=list(test['text']); yt=np.array(test['label'])
    ti,vi=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=SEED)
    write_json(ROOT/'data'/'split_manifest.json',dict(dataset=DATASET,revision=dataset_revision,seed=SEED,train_indices=ti.tolist(),validation_indices=vi.tolist(),test_split='official test, all rows'))
    log(f'Loaded train={len(y)} (selection train={len(ti)}, validation={len(vi)}); untouched test={len(yt)}')
    signature=dict(dataset_revision=dataset_revision,model_revision=model_revision,seed=SEED,max_length=512,experiment_version=1)
    signature_path=ROOT/'reports'/'run_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise RuntimeError('Saved experiment signature differs. Move reports/selection.json, reports/transformer_probabilities.npy and models/baseline.joblib to an archive before starting a new experiment.')
    write_json(signature_path,signature)
    selection_path=ROOT/'reports'/'selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text()); candidates=selection['candidates']; selected=selection['selected']; log('Resuming saved validation selection')
    else:
        progress=ROOT/'reports'/'validation_progress.json'
        candidates=json.loads(progress.read_text()) if progress.exists() else []
        for kind,c in [('word',1),('word',4),('word',10),('word_char',4),('word_char',10)]:
            if any(item['features']==kind and item['C']==c for item in candidates): continue
            gc.collect()
            start=time.perf_counter(); estimator=pipeline(kind,c); log(f'Fit candidate features={kind}, C={c}')
            estimator.fit(x[ti].tolist(),y[ti]); prob=estimator.predict_proba(x[vi].tolist())[:,1]
            item=dict(features=kind,C=c,fit_seconds=round(time.perf_counter()-start,3),**scores(y[vi],prob)); candidates.append(item)
            log(f'Validation accuracy={item["accuracy"]:.6f}; F1={item["f1"]:.6f}')
            write_json(ROOT/'reports'/'validation_progress.json',candidates)
            del estimator,prob
            gc.collect()
        selected=max(candidates,key=lambda c:(c['f1'],c['accuracy']))
        write_json(selection_path,dict(candidates=candidates,selected=selected))
    baseline_path=ROOT/'models'/'baseline.joblib'
    if baseline_path.exists(): estimator=joblib.load(baseline_path); log('Loaded saved final baseline')
    else:
        log(f'Refit selected baseline on all {len(y)} official training examples: {selected["features"]}, C={selected["C"]}')
        estimator=pipeline(selected['features'],selected['C']); estimator.fit(x.tolist(),y); joblib.dump(estimator,baseline_path,compress=3)
    # Transform bounded batches: full char-ngram test matrices exceed laptop RAM.
    baseline_cache=ROOT/'reports'/'baseline_probabilities.npy'
    baseline_prob=np.load(baseline_cache) if baseline_cache.exists() else np.concatenate([estimator.predict_proba(xt[offset:offset+128])[:,1] for offset in range(0,len(xt),128)])
    np.save(baseline_cache,baseline_prob)
    baseline=scores(yt,baseline_prob)
    feature_names=estimator.named_steps['tfidf'].get_feature_names_out()
    coefficients=estimator.named_steps['classifier'].coef_[0]
    order=np.argsort(coefficients)
    write_json(ROOT/'reports'/'feature_weights.json',dict(
        negative=[dict(feature=str(feature_names[i]),coefficient=float(coefficients[i])) for i in order[:20]],
        positive=[dict(feature=str(feature_names[i]),coefficient=float(coefficients[i])) for i in order[-20:][::-1]],
        intercept=estimator.named_steps['classifier'].intercept_.tolist(),classes=estimator.classes_.tolist(),
        total_features=len(feature_names),note='Global fitted coefficients, not causal effects or per-review explanations. Feature vector values also affect each prediction.'))
    log(f'BASELINE TEST {json.dumps(baseline)}')
    rows=[dict(id='baseline',name='TF-IDF + Logistic Regression',**baseline)]
    train_count=len(y); validation_count=len(vi); test_fingerprint=test._fingerprint
    del estimator,feature_names,coefficients,order,x,y,train,test,dataset,ti,vi
    gc.collect()
    transformer_prob=None; device_name=None; inference_seconds=None
    if not args.skip_transformer:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        local=ROOT/'models'/'transformer'; local.mkdir(exist_ok=True)
        log(f'Loading transformer with torch {torch.__version__}; CUDA available={torch.cuda.is_available()}')
        source_manifest=local/'source.json'
        reuse_local=any((local/name).exists() for name in ['model.safetensors','pytorch_model.bin']) and source_manifest.exists() and json.loads(source_manifest.read_text()).get('revision')==model_revision
        if reuse_local:
            log('Loading previously saved pinned safetensors checkpoint locally; avoiding memory-heavy reserialization')
            tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True)
            transformer=AutoModelForSequenceClassification.from_pretrained(local,local_files_only=True)
        else:
            if source_manifest.exists() and json.loads(source_manifest.read_text()).get('revision')!=model_revision:
                raise RuntimeError('Local transformer revision differs. Archive models/transformer before downloading a new revision.')
            snapshot_download(MODEL,revision=model_revision,local_dir=local,allow_patterns=['config.json','tokenizer*','vocab.txt','special_tokens_map.json','model.safetensors','pytorch_model.bin'])
            write_json(source_manifest,dict(id=MODEL,revision=model_revision))
            tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True)
            transformer=AutoModelForSequenceClassification.from_pretrained(local,local_files_only=True)
        log(f'Transformer id2label={transformer.config.id2label}; IMDb labels 0=negative,1=positive')
        device='cuda' if torch.cuda.is_available() else 'cpu'; device_name=torch.cuda.get_device_name(0) if device=='cuda' else platform.processor()
        # Half precision on the 4 GB laptop GPU keeps attention buffers bounded.
        transformer.to(device=device,dtype=torch.float16 if device=='cuda' else torch.float32).eval()
        torch.set_num_threads(min(8,os.cpu_count() or 4))
        cache=ROOT/'reports'/'transformer_probabilities.npy'; start=time.perf_counter()
        probabilities=np.load(cache).tolist() if cache.exists() else []
        # Dataset labels and model classifier labels both map 0=negative, 1=positive.
        with torch.inference_mode():
            for offset in range(len(probabilities),len(xt),4):
                batch=tokenizer(xt[offset:offset+4],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                result=torch.softmax(transformer(**batch).logits.float(),dim=-1)[:,1].cpu().numpy()
                probabilities.extend(result.astype(float).tolist())
                if offset%400==0 or len(probabilities)==len(xt):
                    np.save(cache,np.array(probabilities)); log(f'Transformer evaluated {len(probabilities)}/{len(xt)} reviews on {device}')
        inference_seconds=round(time.perf_counter()-start,3); transformer_prob=np.array(probabilities)
        transformer_scores=scores(yt,transformer_prob); rows.append(dict(id='transformer',name='DistilBERT IMDb',**transformer_scores)); log(f'TRANSFORMER TEST {json.dumps(transformer_scores)}')
    metrics=dict(generated_at=datetime.now(timezone.utc).isoformat(),dataset=dict(id=DATASET,revision=dataset_revision,train_count=train_count,test_count=len(yt),validation_count=validation_count,labels=['negative','positive'],test_fingerprint=test_fingerprint),
        models=rows,training=dict(seed=SEED,candidates=candidates,selected=selected,refit_train_count=train_count,positive_label=1,threshold=.5,transformer_id=MODEL,transformer_revision=model_revision,transformer_max_length=512,transformer_batch_size=4,transformer_dtype='float16 on CUDA; float32 on CPU; float32 softmax',transformer_device=device_name,transformer_inference_seconds=inference_seconds,notes=[
        'Hyperparameters selected by positive-class F1 on stratified validation from official training data only.',
        'Final baseline refitted on all 25000 labeled training rows. Both final models evaluated on the same 25000 official test rows.',
        'The pinned upstream training notebook uses load_dataset("imdb"), trains on ds["train"] for one epoch and evaluates on ds["test"]. Both use the same named official train/test splits; the historical upstream dataset revision was not pinned, so byte-identical training data cannot be independently guaranteed.',
        'The upstream author evaluated on the official test split during training reporting. This is an established benchmark comparison, not a fresh external blind test. The notebook shows no test gradient updates or test-based early stopping.',
        'Confidence is uncalibrated model probability, not a guarantee of correctness. Positive-class binary precision, recall and F1 are reported.',
        'No stopword removal preserves negation. Sublinear TF reduces repeated-word dominance. Word bigrams preserve short phrases; char features, if selected, add spelling and morphology evidence. Feature branch weights are fixed heuristic choices; candidate feature types and C are selected on validation. No individual design choice is proven beneficial without a matched ablation.'
        ]))
    write_json(ROOT/'reports'/'metrics.json',metrics); plot_reports(metrics)
    import runpy
    runpy.run_path(str(ROOT/'training'/'render_validation.py'))
    np.savez_compressed(ROOT/'reports'/'predictions.npz',labels=yt,baseline_probabilities=baseline_prob,transformer_probabilities=transformer_prob)
    samples=[]
    for i in range(0,len(xt),250):
        samples.append(dict(index=i,text=xt[i],label=int(yt[i]),baseline_probability=float(baseline_prob[i]),transformer_probability=float(transformer_prob[i]) if transformer_prob is not None else None))
    write_json(ROOT/'reports'/'paired_examples.json',samples)
    write_json(ROOT/'data'/'provenance.json',dict(dataset=DATASET,dataset_revision=dataset_revision,model=MODEL,model_revision=model_revision,source_urls=[f'https://huggingface.co/datasets/{DATASET}',f'https://huggingface.co/{MODEL}'],raw_reviews_redistributed='100 labeled test review examples in reports/paired_examples.json; full dataset downloaded locally only'))
    runpy.run_path(str(ROOT/'training'/'write_manifest.py'))
    log('Experiment complete. Saved metrics, confusion matrices, probabilities, fitted baseline and transformer.')

if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        (ROOT/'reports').mkdir(exist_ok=True)
        with (ROOT/'reports'/'training.log').open('a',encoding='utf-8') as handle:
            handle.write(traceback.format_exc())
        raise
