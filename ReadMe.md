# Restaurant Review Sentiment Analysis App

A Streamlit application that uses a fine-tuned DistilBERT model to analyze the sentiment of restaurant reviews. This app can load models from Hugging Face Hub or local directories and classify reviews as Positive or Negative.

## Features

- **Model Loading**: Load fine-tuned DistilBERT models from Hugging Face Hub or local paths
- **Multiple Input Methods**: 
  - Single review analysis
  - Batch processing of multiple reviews
  - CSV file upload for bulk analysis
- **Interactive Interface**: User-friendly web interface with real-time predictions
- **Visualization**: Confidence score charts and summary statistics
- **Export Results**: Download analysis results as CSV files

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Load a Model**:
   - Choose between Hugging Face Hub or Local Path
   - For Hugging Face Hub: 
     Model URLs: https://huggingface.co/RinInori/distilbert-restaurant-sentiment

     from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

     model = DistilBertForSequenceClassification.from_pretrained("RinInori/distilbert-restaurant-sentiment")

     tokenizer = DistilBertTokenizer.from_pretrained("RinInori/distilbert-restaurant-sentiment" )

7.6 Public Model Usage Example
This demonstrates how anyone can load and use uploaded model with just two lines of code.

This makes model easily accessible to the machine learning community.

Anyone can use my model with:

[ ]
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("RinInori/distilbert-restaurant-sentiment")

tokenizer = DistilBertTokenizer.from_pretrained("RinInori/distilbert-restaurant-sentiment" )

   - For Local Path: Enter full path to your fine-tuned model directory
   - Click "Load Model"

2. **Analyze Reviews**:
   - **Single Review**: Enter text and click "Analyze Sentiment"
   - **Multiple Reviews**: Enter multiple reviews (one per line)
   - **Upload File**: Upload a CSV file with a 'review' column

## Model Requirements

The app expects models to be compatible with the Hugging Face Transformers library:

- **Architecture**: DistilBERT for sequence classification
- **Labels**: Binary classification (0: Negative, 1: Positive)
- **Format**: Standard Hugging Face model format with `config.json`, `pytorch_model.bin`, and tokenizer files

## Example Model Loading

### From Hugging Face Hub
```
Model name: distilbert-base-uncased
```

### From Local Directory
```
Model path: /path/to/your/fine-tuned-distilbert
```

The local directory should contain:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`

## Sample Usage

1. Load the base DistilBERT model for testing
2. Try the sample reviews provided in the sidebar
3. Upload your own CSV file with restaurant reviews
4. Analyze sentiment and download results

## Technical Details

- **Framework**: Streamlit for the web interface
- **Model**: DistilBERT (Distilled BERT) for efficient inference
- **Tokenization**: Maximum sequence length of 128 tokens (configurable)
- **Output**: Sentiment classification with confidence scores
- **Visualization**: Plotly charts for confidence visualization

## File Structure

```
sentiment_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure the model path is correct
   - Check internet connection for Hugging Face Hub models
   - Verify model compatibility with transformers library

2. **Memory Issues**:
   - Reduce batch size for large files
   - Use CPU instead of GPU if memory is limited

3. **Tokenization Errors**:
   - Ensure text is properly formatted
   - Check for special characters that might cause issues

### Performance Tips

- Use smaller models for faster inference
- Enable batch processing for multiple reviews
- Adjust max_length parameter based on your review lengths

## Based on Fine-Tuning Notebook

This application is designed to work with models trained using the provided fine-tuning notebook:
- **Dataset**: European Restaurant Reviews
- **Architecture**: DistilBERT with sequence classification head
- **Training**: Fine-tuned with class weighting and early stopping
- **Evaluation**: Comprehensive metrics and overfitting prevention

## License

This project is open source and available under the MIT License.

