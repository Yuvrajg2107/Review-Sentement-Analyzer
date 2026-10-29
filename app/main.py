"""CPU inference API. Both outputs are computed from the submitted review."""
from pathlib import Path
from functools import lru_cache
from threading import Lock
import json
import logging
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Review Room", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static", check_dir=False), name="static")
app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")
INFERENCE_LOCK = Lock()
LOG = logging.getLogger(__name__)


class Review(BaseModel):
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a review containing text.")
        return value


@lru_cache(maxsize=1)
def load_models():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    baseline = joblib.load(ROOT / "models" / "baseline.joblib")
    folder = ROOT / "models" / "onnx"
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    transformer = ort.InferenceSession(str(folder / "model.onnx"),
                                      sess_options=options, providers=["CPUExecutionProvider"])
    return baseline, tokenizer, transformer


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/status")
def status():
    baseline = (ROOT / "models" / "baseline.joblib").exists()
    folder = ROOT / "models" / "onnx"
    transformer = all((folder / name).is_file() for name in
                      ("model.onnx", "tokenizer.json"))
    return {"ready": baseline and transformer, "baseline_ready": baseline,
            "transformer_ready": transformer, "report_ready": (ROOT / "reports" / "metrics.json").exists()}


@app.get("/api/report")
def report():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "Evaluation is still running. Metrics appear after completion.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/features")
def features():
    path = ROOT / "reports" / "feature_weights.json"
    if not path.exists():
        raise HTTPException(503, "Feature coefficients are not yet exported.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/training-log", response_class=PlainTextResponse)
def training_log():
    path = ROOT / "reports" / "training.log"
    if not path.exists():
        raise HTTPException(503, "Training log is not yet available.")
    return path.read_text(encoding="utf-8")


@app.post("/api/predict")
def predict(review: Review):
    try:
        with INFERENCE_LOCK:
            baseline, tokenizer, transformer = load_models()
            start = time.perf_counter()
            probabilities = baseline.predict_proba([review.text])[0]
            # Map using classifier classes, rather than assuming class-column order.
            classes = list(baseline.classes_)
            negative, positive = float(probabilities[classes.index(0)]), float(probabilities[classes.index(1)])
            baseline_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            encoded = tokenizer.encode(review.text)
            inputs = {"input_ids": np.asarray([encoded.ids], dtype=np.int64),
                      "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64)}
            logits = transformer.run(["logits"], inputs)[0][0]
            scores = np.exp(logits - np.max(logits))
            scores = (scores / scores.sum()).tolist()
            transformer_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        LOG.exception("Inference unavailable")
        raise HTTPException(503, "Models could not load. Check the deployment artifacts and server log.") from exc

    def result(model_id, name, neg, pos, milliseconds):
        return {"id": model_id, "name": name, "label": "positive" if pos >= neg else "negative",
                "confidence": max(neg, pos), "probabilities": {"negative": neg, "positive": pos},
                "latency_ms": round(milliseconds, 2)}

    return {"predictions": [result("baseline", "TF-IDF + Logistic Regression", negative, positive, baseline_ms),
                            result("transformer", "DistilBERT · IMDb (ONNX)", scores[0], scores[1], transformer_ms)],
            "disclaimer": "Confidence is a model probability, not measured correctness. DistilBERT uses the first 512 tokens; the baseline reads the full review. Live predictions use an compressed ONNX export; saved benchmark scores measure the original PyTorch model."}
