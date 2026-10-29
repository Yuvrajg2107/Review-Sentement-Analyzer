# Explaining Review Room in an interview

## A short opening

“I built a movie-review sentiment comparison. One model learns weighted text features with Logistic Regression. The other is a pretrained transformer from Hugging Face. The app runs both on the same input and shows a separate evaluation on labeled test reviews, so a confident prediction is not mistaken for proven correctness.”

The completed run measured 90.744% accuracy for TF-IDF + Logistic Regression and 92.800% for DistilBERT on the same 25,000 test reviews. DistilBERT made 514 more correct predictions. The classical model's positive-class F1 was 0.907284, compared with 0.927867 for DistilBERT. These are measured results, rather than target scores.

The selected baseline used combined word and character features with `C = 10`. It scored 91.02% accuracy on the training-derived validation split before refitting on all training reviews. Be clear that this validation result and the final test accuracy answer different questions.

## The classical pipeline

```text
Labeled training reviews
        │
        ├── training partition → fit vocabulary and IDF weights
        │                            │
        │                      TF-IDF feature vectors
        │                            │
        │                     Logistic Regression
        │                            │
        └── validation partition → choose configuration
                                     │
                        refit chosen recipe on training split
                                     │
                        evaluate once on official test split
```

The vectorizer must be fitted using training text only. Fitting its vocabulary or IDF values on test reviews would let information from the evaluation set enter training.

TF-IDF increases the weight of terms that occur in a review but are less common across the training collection. The implementation's exact vectorizer parameters, including smoothing and normalization, belong in the saved training configuration.

This vectorizer uses sublinear term frequency: a term appearing `tf` times receives `1 + log(tf)` before IDF weighting. Repetition therefore has less influence than a raw count. With smoothing, IDF is `log((1 + N) / (1 + df)) + 1`, where `N` is the number of training reviews and `df` counts reviews containing the term. Each vectorizer then applies L2 normalization. These are established TF-IDF options.

Logistic Regression learns a weight for each feature and an intercept. For a binary classifier:

```text
z = w · x + b
P(positive | x) = 1 / (1 + exp(-z))
P(negative | x) = 1 - P(positive | x)
```

The weights are learned from labeled examples. They are not manually chosen sentiment scores. Regularization limits coefficient magnitude to reduce overfitting. In scikit-learn, a larger `C` means weaker regularization.

The implemented classifier uses L2 regularization and no class reweighting. The labeled IMDb splits are balanced. Its optional character branch has a fixed feature weight of 0.5 against the word branch's 1.0. That branch weight scales the input features. It is different from the classifier coefficients learned during training.

The website also exposes actual learned feature coefficients through `/api/features`. Positive coefficients push the classifier toward positive sentiment, while negative coefficients push toward negative sentiment. Their effect still depends on the TF-IDF feature value and the rest of the review. A coefficient is an association learned from this dataset, not a rule that a phrase must always carry one sentiment.

## Why feature choices matter

Word n-grams can capture phrases such as “not good” that differ from the meaning of an isolated word. Character n-grams can capture useful fragments and some spelling variation. They also increase memory usage and may learn accidental correlations. Their value needs validation rather than assumption.

Cleaning must preserve evidence needed for sentiment. Removing every “not” or stripping all punctuation can erase meaning. A text classifier still has limited handling of long-range negation and sarcasm even with richer features.

Describe only the feature branches and tuning choices actually saved in the final training configuration. A validation score improvement shows that a choice helped on that validation sample; it does not establish a new general-purpose technique.

## The transformer pipeline

```text
Review → tokenizer → token IDs and attention mask
       → pretrained transformer → sentiment logits → softmax probabilities
```

The tokenizer and model checkpoint must come from the same repository revision. The model's label mapping determines which output is positive or negative; never assume output index zero is positive.

A transformer's attention layers combine information across tokens. The checkpoint has already learned language representations before sentiment fine-tuning. That extra pretraining is a real difference from the classical model and should be disclosed when discussing “the same dataset.”

Transformers have an input length limit. The evaluation and the app must disclose their truncation policy. A long review can contain decisive sentiment beyond the retained tokens.

## Reading the results

Treat positive sentiment as the positive class unless the report states otherwise. A confusion matrix uses actual labels as rows and predicted labels as columns, ordered negative then positive:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | True negatives (TN) | False positives (FP) |
| Actual positive | False negatives (FN) | True positives (TP) |

| Metric | Formula | Interpretation |
|---|---|---|
| Accuracy | `(TP + TN) / all examples` | Fraction of predictions matching the labels |
| Precision | `TP / (TP + FP)` | Among predicted positives, fraction labeled positive |
| Recall | `TP / (TP + FN)` | Among labeled positives, fraction found |
| F1 | `2 × precision × recall / (precision + recall)` | Balances positive-class precision and recall |

The averaging convention matters. Positive-class F1 and macro F1 are different measurements. Use the convention recorded in the report, and compare like with like.

If two models differ only slightly, one run does not establish a dependable advantage on future data. Compare paired predictions and uncertainty before making a strong superiority claim. Speed and resource use can also matter, but only measured timings should be quoted.

## Questions to expect

**Why Logistic Regression?** It is a strong, explainable text baseline that trains on sparse features and is inexpensive to run. Its learned coefficients offer a way to inspect influential features.

**Why not guarantee that your model wins?** The outcome depends on the labeled data and the comparator. Choosing a weak comparator or tuning on test labels could improve the headline while making the comparison less useful.

**Does confidence prove correctness?** No. Both models can be confidently wrong. Calibration would need its own evaluation with held-out labels.

**What is original here?** The implementation and experiment design. The interface makes the predictions and supporting evidence inspectable. Combining established features and a standard classifier is engineering work, not evidence of a new machine-learning architecture.

**What would you improve next?** Inspect errors on a separate development set, test product reviews separately, measure calibration, and consider how long reviews should be handled. Keep the final evaluation set out of the tuning loop.

## Demonstration sequence

1. Open the comparison tab and submit one clear positive review and one clear negative review.
2. Try an ambiguous or mixed review. Explain why disagreement is useful evidence of uncertainty.
3. Open evaluation reports. State the test sample size, metric definitions, and measured winner.
4. Show the architecture and actual training evidence. Explain feature learning, regularization, and the split policy.
5. Finish with the limitations that affect the intended use: movie-review domain, input truncation, and confidence interpretation.
