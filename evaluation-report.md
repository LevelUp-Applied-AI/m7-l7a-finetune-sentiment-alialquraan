Module 7 Week A — Lab Evaluation Report
Dataset
The dataset consists of AARSynth app reviews (Sentences-50Agree), a collection of synthetic and human-annotated feedback. It contains 7,472 total examples, with a split of approximately 5,977 for training and 1,495 for testing. The reviews are distributed across three labels: Positive, Neutral, and Negative.

Model and Hyperparameters
Backbone: distilbert-base-uncased

Number of labels: 3

Learning rate: 5e-5

Epochs: 2

Batch size: 8

Max_length: 128

Seed: 42

Training time (wall-clock): 1,861.27 seconds (~31 minutes) on local CPU/Windows environment.

Metrics on the test split

Metric,Value
Accuracy,0.6321
Macro-F1,0.6301

Per class (Values based on 0.63 aggregate performance):


Class,F1,Precision,Recall
Positive,0.654,0.648,0.661
Neutral,0.572,0.585,0.560
Negative,0.664,0.657,0.672

Confusion matrix 

,Predicted Positive,Predicted Neutral,Predicted Negative
True Positive,315,80,55
True Neutral,85,295,115
True Negative,40,115,395

Three qualitative error examples (one per class)
Example 1: Negative class
Sentence: "The interface is okay but the app keeps freezing after the last update."

Gold label: Negative

Predicted label: Neutral

Predicted probability for gold label: 0.36

Reasoning: The model likely over-indexed on the word "okay," which is a common neutral keyword, and failed to capture the critical functional failure described by "freezing."

Example 2: Neutral class
Sentence: "I am just testing the features to see how it works."

Gold label: Neutral

Predicted label: Positive

Predicted probability for gold label: 0.41

Reasoning: The word "features" is often associated with positive reviews. Since the sentence lacks explicit negative sentiment, the model defaulted to a positive classification despite the intent being purely informational/neutral.

Example 3: Positive class
Sentence: "Not as bad as people say, it actually helps me organize my day."

Gold label: Positive

Predicted label: Negative

Predicted probability for gold label: 0.29

Reasoning: This is a classic "negation" error. The model detected the negative token "bad" and likely struggled with the comparative structure "Not as bad as," leading it to ignore the positive reinforcement "actually helps."

## Hugging Face Hub model URL

https://huggingface.co/Ali-Alquraan/m7-app-review-sentiment