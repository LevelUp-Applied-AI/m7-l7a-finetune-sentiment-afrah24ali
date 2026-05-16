# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 1 | 1 | 100% |
| lexical_trigger | 1 | 1 | 100% |
| domain_shift | 0 | 1 | 0% |
| length_extreme | 0 | 0 | N/A |
| sarcasm | 0 | 0 | N/A |
| other | 0 | 0 | N/A |

## Confirmed hypotheses

The model failed on "domain_shift".Row 3 was expected to be "neutral" because it is sport news rather than an app review , but the model predicted "negative" with probability 0.4464 this supports the hypothesis that classifier may struggle when the input is outside the app-review domain it was trained on .


## Refuted hypotheses

The model handled the "negation" and "lexical_trigger" examples better than expected.Row 1 included the positive word "improve" inside a neagative sentence ,but the model predicted "negative" with probability 0.8568.Rwo 2 included the positive cue "reliable" after "no longer",and the model also predicted "negative" with probability 0.9555.

## What the results reveal about the decision boundary

The results suggest that the model is not only matching isolated positive words.I t correctly used context around negation phrases like "did not improve" and "no longer reliable .
However,the domain-shift miss suggests that the decision boundary is still strongly by patterns from app-review training data. When the text did not sound like an app review, the model did not confidently assign "neutral; instead ,it leanded slightly negative. 
