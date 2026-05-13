## Module 7 week A - LAB Evaluation Report
# Dataset
The dataset contains 7,472 app reviews collected from 9 different mobile applications.
Each review is labeled into one of three sentiment classes :negative (0),neutral (1), and positive(2).

The dataset was split into:
-Training set :80%
-Test set 20%
-Split seed :42

Class distribution is approximately balanced,but neutral reviews are slightly harder to classify due to ambiguity in wording.

# Model and hyperparameters 
-Backborn :distilbert-base-uncased
-Number of labels:3
-Epochs:2
-Batch size:8
-Max sequence lentgh :128
-Seed:42
Training time: ~16 minutes on local CPU/GPU environment.

# Matrics on the test split
Accuracy: 0.6314
Macro-F1: 0.6298

# per-class performance
"per_class_f1": {
    "negative": 0.7051671732522796,
    "neutral": 0.4850976361767729,
    "positive": 0.6990291262135923
  },
  "per_class_precision": {
    "negative": 0.7131147540983607,
    "neutral": 0.4627450980392157,
    "positive": 0.7243460764587525
  },
  "per_class_recall": {
    "negative": 0.6973947895791583,
    "neutral": 0.509719222462203,
    "positive": 0.6754221388367729
  }
# Confusion matrix
The model performance best on positive and negative classes but struggles with neutral reviews.
Most common errors :
Neutral >> Negative misclassification
Neutral >> Positive misclassification

# Three qualitative error examples
# Example 1
- Text: it was okay at the start and got me focussed but my phone's screen timed out and locked and then my tree withered. please fix this 
- Gold label: negative  
- Predicted label: neutral  
- Gold probability: 0.754978716373443
Short ambiguous reviews are harder for the model because there is not enough context to strongly determine sentiment.

#  Example 2
- Text:its a good app but we dont have a night mode option for reading in which the background becomes black and text becomes white in color.  
- Gold label: positive  
- Predicted label: neutral  
- Gold probability: 0.799402773380279
The model failed because the sentence includes both positive and neutral phrases, causing diluted confidence.

# Example 3
- Text:  i have lots of state that i watch. my home town will not show what it feels like. all the others do. i have tried a few different things and it won't work. don't understand but it is important to me to know how cold it feels out. i can go to hourly and it tells me but should not have to! i'm uninstalling!
- Gold label: neutral  
- Predicted label: negative  
- Gold probability: 0.771129846572875
The model misclassified this because it contains negative-sounding words even though the overall sentiment is neutral.


## Summary insight
The model confuses neutral reviews most often with both positive and negative classes, indicating that neutrality is underrepresented in strong linguistic signals and is harder for transformer-based classification.

## Hugging Face Hub model URL
https://huggingface.co/afrahali25/m7-app-review-sentiment