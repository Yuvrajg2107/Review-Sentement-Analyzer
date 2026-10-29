"""Download the pinned upstream checkpoint without rerunning the experiment."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))


def main():
    if sys.platform == "win32":
        import truststore
        truststore.inject_into_ssl()
    from huggingface_hub import snapshot_download
    report = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    config = report["training"]
    model_id, revision = config["transformer_id"], config["transformer_revision"]
    folder = ROOT / "models" / "transformer"
    folder.mkdir(parents=True, exist_ok=True)
    source_path = folder / "source.json"
    source = {"id": model_id, "revision": revision}
    if source_path.exists() and json.loads(source_path.read_text(encoding="utf-8")) != source:
        raise RuntimeError("Archive models/transformer before switching to another checkpoint revision.")
    snapshot_download(repo_id=model_id, revision=revision, local_dir=folder,
                      allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin",
                                      "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "vocab.txt"])
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Downloaded {model_id}@{revision} to {folder}")


if __name__ == "__main__":
    main()
