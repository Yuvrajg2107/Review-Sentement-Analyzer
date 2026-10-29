# Serving model

This is a compressed export of `lvwerra/distilbert-imdb` at revision
`0fc02cd68445b599a9cb2da2368050e7fb31d29a`, whose model card declares
Apache-2.0. The original model card is preserved in
`reports/upstream_model_card.md`.

`training/export_onnx.py` exports the original classifier, stores embeddings in
FP16 with FP32 computation, and dynamically quantizes linear weights to UINT8.
Unsigned weights avoid U8/S8 accumulation saturation on AVX2 processors.
The tokenizer is copied from the same checkpoint. No labels or model weights
are retrained during export.

`manifest.json` records provenance, checksum, size and a comparison with the
original model on the saved paired examples plus five additional inputs.
That sample comparison is not a full test-set accuracy measurement. The saved
benchmark report still describes the original PyTorch model.
