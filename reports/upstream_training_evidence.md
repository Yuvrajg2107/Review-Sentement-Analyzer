# Upstream training evidence

Source: https://huggingface.co/lvwerra/distilbert-imdb/blob/0fc02cd68445b599a9cb2da2368050e7fb31d29a/distilbert-imdb-training.ipynb

The pinned notebook contains these training steps:

```python
ds = load_dataset("imdb")
def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True)
    return outputs
training_args = TrainingArguments(num_train_epochs=1, ...,
                                  evaluation_strategy="epoch")
trainer = Trainer(...,
                  train_dataset=tokenized_ds["train"],
                  eval_dataset=tokenized_ds["test"], ...)
trainer.train()
```

This documents official IMDb training and test splits without subsampling. The original dataset revision was not pinned. The model card reports 0.928 test accuracy, which is upstream evidence rather than a result measured by this project. This project independently evaluates the frozen checkpoint. The upstream notebook shows no test gradient updates or test-based early stopping. Evaluation on an established public benchmark is weaker evidence than a fresh external test.
