---
license: apache-2.0
language:
- en
metrics:
- accuracy
- f1
- precision
- recall
base_model:
- distilbert/distilbert-base-uncased
pipeline_tag: text-classification
---
Model Card for RinInori/distilbert-restaurant-sentiment

A Streamlit application that uses a fine-tuned DistilBERT model to analyze the sentiment of restaurant reviews. This app can load models from Hugging Face Hub or local directories and classify reviews as Positive or Negative. App link: https://finetuningsentimentanalysisrestaurantreview-ezvyaqlpzg4d8kxvjj.streamlit.app/

### Model Description
This is a fine-tuned DistilBERT model for sentiment analysis of restaurant reviews. 
It is designed to classify reviews as either positive or negative. The model was developed as part of a project demonstrating the process of fine-tuning a DistilBERT model using the European Restaurant Reviews dataset, including data loading, model evaluation, handling class imbalance, and preventing overfitting.

Model type: DistilBERTForSequenceClassification

Finetuned from model: distilbert-base-uncased

### Model Sources
Repository: https://huggingface.co/RinInori/distilbert-restaurant-sentiment

## Uses

### Direct Use
This model can be directly used for classifying the sentiment of restaurant reviews as either positive or negative. 
It is particularly suited for analyzing text from food/restaurant reviews, given its training data.

Anyone can use this model with:
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("RinInori/distilbert-restaurant-sentiment")
tokenizer = DistilBertTokenizer.from_pretrained("RinInori/distilbert-restaurant-sentiment" )
```

### Downstream Use [optional]
This model can be integrated into larger applications requiring sentiment analysis of customer feedback in the restaurant industry, such as customer service analytics platforms, review aggregation services, or market research tools.

### Out-of-Scope Use
This model is specifically fine-tuned for restaurant review sentiment analysis. Using it for sentiment analysis in other domains (e.g., product reviews, movie reviews, social media posts) without further fine-tuning may lead to inaccurate results. It is also not intended for generating text or performing other NLP tasks beyond sentiment classification.

## Bias, Risks, and Limitations
This model was trained on European restaurant reviews. Therefore, it may exhibit biases present in the training data, such as specific linguistic nuances, cultural references, or common sentiments expressed in European contexts. Its performance might degrade when applied to reviews from vastly different cultural or linguistic backgrounds. The model classifies sentiment into two categories (positive/negative) and does not capture nuanced or mixed sentiments. Overfitting was addressed during training using techniques like early stopping and dropout regularization, but it's always a potential risk with fine-tuned models.

## Recommendations
Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model
Use the code below to get started with the model.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("RinInori/distilbert-restaurant-sentiment")
tokenizer = DistilBertTokenizer.from_pretrained("RinInori/distilbert-restaurant-sentiment")

# Example usage:
text = "The food was absolutely delicious and the service was outstanding!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get predicted class and confidence
predicted_class = torch.argmax(predictions, dim=-1).item()
sentiment = "Positive" if predicted_class == 1 else "Negative"
confidence = predictions[0][predicted_class].item()

print(f"Review: {text}")
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
```

## Training Details

### Training Data
The model was fine-tuned on the European Restaurant Reviews dataset from Kaggle. This dataset contains customer reviews for various restaurants, divided into positive and negative sentiments.

**Dataset URL**: https://www.kaggle.com/datasets/gorororororo23/european-restaurant-reviews

### Training Procedure

#### Preprocessing [optional]
Preprocessing steps included tokenization using `DistilBertTokenizer`. The dataset was split into training and validation sets. Class imbalance was addressed using class weighting, and overfitting was prevented using early stopping and dropout regularization.

#### Training Hyperparameters
Training regime: The model was trained using the Hugging Face `Trainer` API. Key techniques included class weighting to handle imbalance, early stopping to prevent overfitting, and dropout regularization.

## Evaluation

### Testing Data, Factors & Metrics
Testing Data: The model's performance was evaluated on a held-out test set from the European Restaurant Reviews dataset.

Factors: Sentiment classification (Positive/Negative).

Metrics: The evaluation included standard classification metrics such as accuracy, precision, recall, and F1-score. 
Confusion matrices were also used to visualize performance.

### Results
Fine-Tuned the model with dataset made the model performance become better.

## Summary

### Model Examination [optional]
More Information Needed

## Environmental Impact
Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019).

Hardware Type: GPU, A100
Cloud Provider: Google Colab (implied by notebook)

## Technical Specifications [optional]

### Model Architecture and Objective
The model uses the DistilBERT architecture, which is a distilled version of BERT, making it smaller and faster while retaining much of BERT's language capabilities. 
The objective was to fine-tune this architecture for binary sentiment classification (positive/negative) on restaurant reviews.

### Compute Infrastructure
Training was performed in a Google Colab environment.

### Software
Python, PyTorch, Hugging Face Transformers library, pandas, numpy, scikit-learn, matplotlib, seaborn, nltk.

## Citation 
BibTeX:

```bibtex
@misc{rin_inori_distilbert_restaurant_sentiment,
  author = {RinInori},
  title = {distilbert-restaurant-sentiment},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/RinInori/distilbert-restaurant-sentiment}}
}
```

APA:

RinInori. (2024). *distilbert-restaurant-sentiment*. Hugging Face Hub. Retrieved from https://huggingface.co/RinInori/distilbert-restaurant-sentiment
