# Task-to-file map

| Assignment requirement | Implementation or evidence |
|---|---|
| TF-IDF with Logistic Regression baseline | `training/run_experiment.py`, fitted `models/baseline.joblib` |
| Comparison with a Hugging Face transformer | `lvwerra/distilbert-imdb`, pinned by the report and fetched by `training/download_transformer.py` |
| Accuracy, precision, recall and F1 | `reports/metrics.json`, shown in the website's evaluation tab |
| Confusion matrix | `reports/baseline_confusion_matrix.png`, `reports/transformer_confusion_matrix.png` and the evaluation tab |
| Review text input and both predictions | `app/main.py` and `app/static/` |
| Sentiment and confidence from both models | `/api/predict`, displayed in the compare tab |
| Organized data and notebook folders | `data/`, `notebooks/`, `app/`, `requirements.txt` |
| README with setup plus screenshots and a table of results | `README.md` |
| Working demo | Local FastAPI application at `http://127.0.0.1:8000` after setup |
| Architecture and model explanation | Website architecture tab and `docs/interview-guide.md` |
| Actual training evidence | `reports/training.log` and training screenshots under `reports/screenshots/` |
| Dataset provenance | `data/provenance.json`, `data/split_manifest.json`, `docs/methodology.md` |
| Measured environment | `reports/environment.json` |
| Learned feature coefficients | `reports/feature_weights.json`, `/api/features` |

The setup supports a local demo, which is one of the assignment's accepted delivery options. A hosted deployment is a separate operational step.

The saved result files establish what was measured. A path in this map identifies where an artifact belongs, and does not replace checking that the final file exists and matches the report.
