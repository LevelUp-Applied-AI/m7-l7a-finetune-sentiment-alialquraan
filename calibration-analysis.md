# Calibration Analysis

> **TODO** — fill each section after running your manual evaluation and reliability diagram.

## Reliability diagram interpretation

What does your saved diagram (`figures/reliability-diagram.png`) look like? Where is the model over-confident vs. under-confident? Cite specific bucket values.

The reliability diagram shows that the model is somewhat calibrated at high confidence levels, but it becomes noticeably over-confident in several mid-confidence ranges. In the lower-confidence buckets (0.0–0.3), there were almost no predictions, which indicates that the model usually assigns moderate-to-high confidence scores rather than uncertain predictions.

The 0.3–0.4 confidence bucket achieved an empirical accuracy of about 0.33, which is close to its confidence level. However, the 0.5–0.6 bucket had an accuracy of only about 0.44, despite predictions being made with around 0.55 confidence on average. Similarly, the 0.6–0.7 bucket achieved only 0.50 accuracy, showing that the model was over-confident in this range.

The highest-confidence bucket (0.9–1.0) performed much better, with an empirical accuracy of about 0.86. This indicates that highly confident predictions are usually correct, although the model still slightly overestimates its certainty even at very high confidence levels.

Overall, the diagram suggests that the model’s confidence estimates are not perfectly aligned with real-world correctness, especially in the medium-confidence region.

## Expected Calibration Error

Report your ECE. Interpret what it says about model trustworthiness for production use.

The model achieved an Expected Calibration Error (ECE) of 0.1124.

This means that, on average, the model’s predicted confidence differs from its actual accuracy by about 11%. An ECE of 0 would indicate perfect calibration, so this result shows moderate calibration quality but still leaves room for improvement.

For production use, this means the model’s probability scores should not be interpreted as perfectly reliable confidence estimates. For example, when the model predicts with 70% confidence, the true accuracy may be substantially lower. While the model is usable for inference tasks, additional calibration techniques would improve trustworthiness in applications where confidence scores influence decisions.

## A specific calibration pattern

Identify one specific pattern (over-confidence on majority class, under-confidence near boundaries, etc.) and reason about why it arose given how the model was trained.

One clear pattern is over-confidence in the medium-confidence buckets (roughly 0.5–0.8). For example, the 0.6–0.7 confidence bucket achieved only about 0.50 empirical accuracy. This suggests the model is more certain about its predictions than it should be.

This pattern likely arose because the DistilBERT classifier was fine-tuned using standard cross-entropy loss, which optimizes classification accuracy rather than calibration quality. Neural networks trained with cross-entropy commonly produce overly confident softmax probabilities, especially when the dataset contains ambiguous examples or overlapping sentiment classes such as neutral versus positive reviews.

Another contributing factor is that app-review sentiment often contains mixed emotional cues. A review may contain both positive and negative wording, causing the model to confidently predict one dominant sentiment even when the example is inherently ambiguous.

## A proposed engineering action

What would you change in production based on these findings? (Threshold-based abstention, temperature scaling, bucket-specific data collection, etc.)

One practical production improvement would be to apply temperature scaling after training. Temperature scaling is a lightweight post-processing calibration method that adjusts softmax confidence values without changing the predicted class labels. This would likely reduce the over-confidence observed in the middle confidence buckets and lower the ECE.

Additionally, I would introduce a confidence threshold for deployment. For example, predictions with confidence below 0.6 could be flagged for human review or treated as uncertain instead of being accepted automatically. This would reduce the risk of incorrect predictions being presented with misleadingly high confidence.

Finally, collecting additional training examples focused on ambiguous neutral reviews could help improve calibration near class boundaries, especially between neutral and positive sentiment.
