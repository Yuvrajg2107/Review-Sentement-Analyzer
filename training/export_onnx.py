"""Export the pinned local DistilBERT checkpoint for CPU-only serving.

Run after download_transformer.py with requirements-training.txt installed.
The saved benchmark report remains a measurement of the original PyTorch model.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    source = ROOT / "models" / "transformer"
    target = ROOT / "models" / "onnx"
    target.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / ".tmp" / "export"
    temporary.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, local_files_only=True, attn_implementation="eager"
    ).cpu().float().eval()

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    example = tokenizer("A movie worth watching.", return_tensors="pt")
    fp32 = temporary / "model.onnx"
    torch.onnx.export(
        Classifier().eval(), (example["input_ids"], example["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "sequence"}
                      for name in ("input_ids", "attention_mask")}
        | {"logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # Export restores the wrapper's training mode; keep dropout disabled for
    # the original-model comparison as well as the exported graph.
    model.eval()
    # Preserve embeddings in FP16 instead of INT8: quantizing the embedding
    # table caused avoidable prediction drift. Cast back to FP32 for compute.
    graph = onnx.load(str(fp32))
    embedding_names = {node.input[0] for node in graph.graph.node if node.op_type == "Gather"}
    casts = []
    for weight in graph.graph.initializer:
        if weight.name not in embedding_names or weight.data_type != TensorProto.FLOAT:
            continue
        name = weight.name
        restored = name + ".fp32"
        weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight).astype(np.float16), name))
        for node in graph.graph.node:
            for i, value in enumerate(node.input):
                if value == name:
                    node.input[i] = restored
        casts.append(helper.make_node("Cast", [name], [restored], to=TensorProto.FLOAT))
    for node in reversed(casts):
        graph.graph.node.insert(0, node)
    mixed = temporary / "embeddings-fp16.onnx"
    onnx.save(graph, str(mixed))
    output = target / "model.onnx"
    # U8/U8 avoids the U8/S8 accumulation saturation seen on AVX2 CPUs.
    quantize_dynamic(str(mixed), str(output), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["MatMul"], per_channel=True,
                     extra_options={"MatMulConstBOnly": True})
    shutil.copyfile(source / "tokenizer.json", target / "tokenizer.json")
    shutil.copyfile(source / "config.json", target / "config.json")

    # Check a fixed, existing sample without changing any benchmark results.
    rows = json.loads((ROOT / "reports" / "paired_examples.json").read_text())
    texts = [row["text"] for row in rows]
    texts += ["A wonderful film. I loved it.", "An awful film. I hated it.",
              "It was okay, but I would not watch it again.", "film " * 700, "x"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    differences, agreements = [], []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            original = model(**inputs).logits.softmax(-1)[0].numpy()
        logits = session.run(None, {name: inputs[name].numpy()
                                   for name in ("input_ids", "attention_mask")})[0][0]
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        differences.append(float(np.max(np.abs(original - scores))))
        agreements.append(bool(original.argmax() == scores.argmax()))
    manifest = {
        "source": json.loads((source / "source.json").read_text()),
        "format": "ONNX: FP16 embedding storage, dynamic per-channel UINT8 MatMul weights, FP32 compute",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "sample_check": {"count": len(texts), "label_agreements": sum(agreements),
                         "max_probability_difference": max(differences),
                         "mean_probability_difference": float(np.mean(differences))},
        "benchmark_note": "reports/metrics.json measures the original PyTorch checkpoint, not this compressed export.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if output.stat().st_size >= 100 * 1024 * 1024:
        raise RuntimeError("Export exceeds GitHub's individual file limit.")
    # This is a compression smoke check, not a claim of benchmark equivalence:
    # require 98% sample label agreement and at most 1.5 percentage points mean drift.
    if sum(agreements) / len(texts) < .98 or np.mean(differences) > .015:
        raise RuntimeError("Export failed the sample parity check; do not publish these artifacts.")


if __name__ == "__main__":
    main()
