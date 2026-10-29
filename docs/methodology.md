# Experiment protocol

## Sources

- Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).
- Transformer: [lvwerra/distilbert-imdb](https://huggingface.co/lvwerra/distilbert-imdb).
- Original dataset description: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The connected Hugging Face repository metadata was inspected when choosing the comparator. It lists the model architecture as DistilBERT and the training dataset as `imdb`. The [upstream training notebook at the pinned model revision](https://huggingface.co/lvwerra/distilbert-imdb/blob/0fc02cd68445b599a9cb2da2368050e7fb31d29a/distilbert-imdb-training.ipynb) loads IMDb's official `train` split for fine-tuning and its `test` split for evaluation, without subsampling. It trains for one epoch and tokenizes to a maximum length of 512. Our final classifier uses those same named splits. The historical dataset revision is not pinned upstream, which limits any claim of identical training bytes.

## Split policy

The classical model uses a stratified validation partition drawn only from the official training split. Feature extraction and classifier fitting for each candidate use that partition's training portion. Validation labels select the configuration. The selected recipe is then refitted on the official training split before final evaluation.

The official test split is reserved for final measurement. Both model predictions must be aligned to the same ordered test rows. Label zero means negative; label one means positive. The transformer label mapping must be checked against its configuration.

The official test split was already used for upstream evaluation. This does not show that test examples were used for gradient updates. It does mean that our comparison reuses an established evaluation set. A separate external dataset would provide a stronger assessment of generalization.

## Candidate selection

Candidate choices are ordinary baseline engineering: TF-IDF feature ranges, optional character features, and Logistic Regression regularization. State the selected values from the recorded run. Do not attribute an improvement to a feature without a matching candidate comparison under the same validation conditions.

The implemented search compares word features at `C = 1, 4, 10` with combined word and character features at `C = 4, 10`. Word features use unigrams and bigrams, sublinear term frequency, Unicode accent normalization, a minimum document frequency of 2, a maximum document frequency of 0.98, and at most 100,000 features. The character branch uses word-boundary character n-grams of lengths 3 through 5, a minimum document frequency of 3, and at most 60,000 features. When present, that branch is weighted by 0.5 relative to the word branch's 1.0. These weights are fixed design choices within this search, rather than separately tuned values.

The classifier uses the `liblinear` solver, a maximum of 500 iterations, and seed 42. Configuration selection maximizes positive-class validation F1, with accuracy as the tie-breaker. The classification threshold is 0.5. These values describe the implemented search; the selected recipe and its measured scores are recorded separately in the final report.

The validation experiment compares candidate configurations. The final test comparison answers a different question: how the chosen local classifier and the upstream transformer perform on these held-out labels.

## Report contents

`reports/metrics.json` is the source for dataset identifiers and revisions, split counts, candidate results, final metrics, confusion matrices, and run details. Training logs and screenshots supplement this machine-readable evidence. A screenshot shows what ran; it is not a substitute for the underlying results and code.

Accuracy, precision, recall and F1 are computed from predictions, not copied from a Hugging Face model card. The confusion matrix is ordered `[negative, positive]` on both axes. All displayed metrics must specify whether they are validation or test results.

The completed test run measured baseline accuracy `0.90744` and transformer accuracy `0.928`. The baseline confusion matrix is `[[11364, 1136], [1178, 11322]]`; the transformer matrix is `[[11623, 877], [923, 11577]]`. These matrices each contain 25,000 predictions and account for the accuracy difference of 514 correctly classified reviews.

Transformer evaluation used GPU FP16 execution, batch size 4, FP32 softmax and a maximum input length of 512 tokens. This describes the measured evaluation configuration. Device and precision choices can cause small differences when reproducing probabilities elsewhere.

## Limits

- IMDb contains movie reviews with strong positive or negative sentiment. It does not establish performance on neutral text, product reviews, or other languages.
- This is binary classification. A probability near the decision threshold can flag uncertainty, but it does not add a trained neutral class.
- The transformer has a token limit and a documented truncation policy. The classical vectorizer uses its own vocabulary and feature limits.
- Classifier probabilities have not automatically been calibrated. Avoid presenting confidence as a verified likelihood of correctness.
- Different pretraining histories prevent a claim that architecture is the only cause of a score difference.
- A single benchmark result does not guarantee an advantage on future reviews.

## Reproduction checklist

Start with a fresh environment, install the documented dependencies, run `training/run_experiment.py`, and inspect the completed report. Compare dataset revision, transformer revision, split seed, split counts, selected parameters and package versions with the recorded run before interpreting numerical differences.

The default source revisions are pinned in the training script. Use `--fresh` to archive previous reusable artifacts under `reports/archive/` and run the experiment again. Omit it to resume a compatible interrupted run. A saved run signature guards against reusing results from a different source revision or experiment configuration.

Some numeric differences can arise across library versions and hardware. Record the differences rather than replacing a measured result with an expected one.

## Resource constraints and retries

The local machine has an RTX 3050 with 4 GB of VRAM and 8 GB of system RAM. Constructing character-feature matrices for the full test collection at once exceeded available memory during development. Batched test inference lowers peak memory without changing the fitted classifier or the evaluation rows.

The training log retains failures and retries. A successful resume can reuse compatible saved validation selection and the fitted baseline. Read the final saved metrics as the authoritative completed evaluation. Do not describe the history as a clean first run.
