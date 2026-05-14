# Adversarial Evaluation Analysis

> **TODO** — fill each section after running `run_adversarial.py` against your fine-tuned model.

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 4 | 5 | 80% |
| lexical_trigger | 2 | 5 | 40% |
| domain_shift | 5 | 6 | 83.3% |
| length_extreme | 4 | 5 | 80% |
| sarcasm | 1 | 5 | 20% |
| other | 1 | 4 | 25% |


Overall accuracy on the adversarial set was 17/30 = 56.7%.

## Confirmed hypotheses

Which categories did the model fail on as you predicted? Cite specific row IDs and predictions.

The sarcasm hypothesis was strongly confirmed.  
Rows 17, 18, 26, and 30 were classified incorrectly.  
The model predicted positive or neutral labels because it focused on positive cue words such as “Fantastic”, “Great”, “Lovely”, and “Wonderful” while missing the sarcastic meaning.

The lexical_trigger hypothesis was also confirmed.  
Rows 7, 9, and 23 failed because the model appeared to overweight positive trigger words such as “excellent” and “reliable” even when the surrounding context was negative.

The other category also exposed weaknesses.  
Rows 19 and 27 containing distorted spellings like “amazzzing” and “goooood” were predicted as negative instead of positive, suggesting poor robustness to noisy text.


## Refuted hypotheses

Which categories did the model handle better than you expected?

The negation hypothesis was mostly refuted.  
The model correctly handled most negation examples including rows 1, 4, 5, and 22.  
This suggests the model learned common negation patterns reasonably well.

The domain_shift hypothesis was weaker than expected.  
The model correctly classified most examples from sports, news, and entertainment domains such as rows 11, 12, 24, and 28.

The length_extreme hypothesis was partially refuted as well.  
The model successfully handled both extremely short and very long examples except for row 29 (“This app.”), which was incorrectly classified as negative instead of neutral.


## What the results reveal about the decision boundary

Articulate one or more specific things the adversarial results say about how the model decides.

The results suggest that the model relies heavily on strong sentiment cue words when making predictions.  
Words such as “excellent”, “reliable”, “great”, and “wonderful” strongly influenced the classifier even when sarcasm or surrounding context reversed the actual meaning.

The model also appears sensitive to clean spelling and standard language patterns.  
Examples with distorted spelling or unusual formatting caused prediction errors even when the sentiment was obvious to humans.

At the same time, the model handled explicit negation patterns relatively well, indicating that negation handling was likely represented in the training data more consistently than sarcasm or noisy text patterns.
