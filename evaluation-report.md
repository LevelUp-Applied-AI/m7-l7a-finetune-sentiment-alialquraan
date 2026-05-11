# Module 7 Week A — Lab Evaluation Report

## Dataset

The dataset used in this lab was the AARSynth app reviews Sentences-50Agree dataset.
It contains 7,472 app reviews collected from 9 different applications with 3 sentiment classes:
negative, neutral, and positive.

The dataset was split internally into:
- Training split: 5,977 examples
- Test split: 1,495 examples

The label mapping used in the model was:
- 0 = negative
- 1 = neutral
- 2 = positive

---

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 5e-5
- Epochs: 2
- Batch size: 8
- Max sequence length: 128
- Random seed: 42
- Training time (wall-clock): approximately 34 minutes on CPU

The model was fine-tuned using Hugging Face Trainer with dynamic padding through `DataCollatorWithPadding`.

---

## Metrics on the test split

### Aggregate Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.6475 |
| Macro-F1 | 0.6441 |

---

### Per-Class Metrics

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Negative | 0.727 | 0.720 | 0.736 |
| Neutral | 0.501 | 0.492 | 0.514 |
| Positive | 0.702 | 0.723 | 0.681 |

> Note: values are rounded for readability.

---

## Confusion matrix

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 367 | 114 | 18 |
| Neutral | 105 | 238 | 120 |
| Positive | 38 | 132 | 363 |

### Analysis

The confusion matrix shows that the model performs best on the negative and positive classes.
The neutral class was the most difficult because many neutral reviews contain mixed or ambiguous sentiment language.

The model frequently confused:
- neutral → positive
- positive → neutral
- negative → neutral

This suggests that the classifier sometimes overweights emotionally charged words even when the overall sentence sentiment is balanced.

---

## Three qualitative error examples (one per class)

### Example 1 — Neutral classified as Negative

- Original sentence:
  "The app works fine but occasionally freezes during startup."

- Gold label:
  Neutral

- Predicted label:
  Negative

- Predicted probability for gold label:
  0.31

### Analysis

The phrase "freezes during startup" likely triggered negative sentiment patterns learned during training.
Although the review includes both positive and negative information, the model focused more heavily on the negative cue phrase.

---

### Example 2 — Positive classified as Neutral

- Original sentence:
  "Great interface and very easy to use for beginners."

- Gold label:
  Positive

- Predicted label:
  Neutral

- Predicted probability for gold label:
  0.42

### Analysis

The sentence expresses positive sentiment but in a mild and factual way.
The model may have interpreted the review as descriptive rather than strongly emotional.

---

### Example 3 — Negative classified as Neutral

- Original sentence:
  "The latest update made the app slower and less reliable."

- Gold label:
  Negative

- Predicted label:
  Neutral

- Predicted probability for gold label:
  0.39

### Analysis

The review contains criticism but does not use extremely emotional language.
The model may have struggled because the sentiment intensity was moderate instead of explicitly negative.

---

## Hugging Face Hub model URL

https://huggingface.co/Ali-Alquraan/m7-app-review-sentiment