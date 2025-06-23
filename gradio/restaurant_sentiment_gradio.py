import gradio as gr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io

# Default model configuration
DEFAULT_MODEL = "RinInori/distilbert-restaurant-sentiment"

# Global variables to store model and tokenizer
model = None
tokenizer = None
model_loaded = False

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #1f77b4 0%, #1f77b4 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 32px rgba(31, 119, 180, 0.3);
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.sub-title {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-bottom: 0;
}

.description {
    font-size: 1rem;
    color: #1e293b;
    margin: 2rem 0;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f0f9ff 0%, #fef3c7 100%);
    border-radius: 12px;
    border-left: 5px solid #1f77b4;
    box-shadow: 0 4px 16px rgba(31, 119, 180, 0.1);
}

.sentiment-positive {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 2px solid #28a745;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(40, 167, 69, 0.2);
}

.sentiment-negative {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 2px solid #dc3545;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(220, 53, 69, 0.2);
}

.confidence-score {
    font-size: 1.3rem;
    font-weight: bold;
    margin-top: 0.5rem;
}

.model-info {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
}

.stats-container {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #9c27b0;
}

.gr-button-primary {
    background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(31, 119, 180, 0.4) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: white !important;
}

.gr-textbox {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus {
    border-color: #1f77b4 !important;
    box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1) !important;
}

.gr-slider {
    accent-color: #1f77b4 !important;
}

.gr-dropdown {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
}

.footer-info {
    text-align: center;
    margin-top: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 12px;
    font-size: 0.9rem;
    color: #64748b;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
"""

def load_model_and_tokenizer(model_name=DEFAULT_MODEL, progress=gr.Progress()):
    """
    Load the fine-tuned DistilBERT model and tokenizer from Hugging Face Hub.
    """
    global model, tokenizer, model_loaded
    
    try:
        progress(0.2, desc="Loading tokenizer...")
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        progress(0.6, desc="Loading model...")
        # Load model
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        model_loaded = True
        
        progress(1.0, desc="Model loaded successfully!")
        
        return f"Model and tokenizer loaded successfully!\n\n **Model:** {model_name}\n**Architecture:** Fine-Tuned DistilBERT\n **Task:** Restaurant Sentiment Classification\n📊 **Classes:** Positive, Negative"
        
    except Exception as e:
        model_loaded = False
        return f"❌ Error loading model: {str(e)}"

def predict_sentiment(text, max_length=128):
    """
    Predict sentiment for a given text using the loaded model.
    """
    global model, tokenizer, model_loaded
    
    if not model_loaded or model is None or tokenizer is None:
        return "❌ Model not loaded. Please load the model first.", None, None
    
    if not text.strip():
        return "⚠️ Please enter some text to analyze.", None, None
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get predicted class and confidence
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        # Get both class probabilities
        negative_prob = predictions[0][0].item()
        positive_prob = predictions[0][1].item()
        
        # Map class to sentiment (0: Negative, 1: Positive)
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        
        # Create result text
        if sentiment == "Positive":
            result_text = f"""
### 😊 Positive Sentiment

**Confidence:** {confidence:.2%}

The model predicts this review expresses **positive** sentiment about the restaurant experience.
"""
        else:
            result_text = f"""
### 😞 Negative Sentiment

**Confidence:** {confidence:.2%}

The model predicts this review expresses **negative** sentiment about the restaurant experience.
"""
        
        # Create confidence chart
        fig = px.bar(
            x=["Negative", "Positive"],
            y=[negative_prob, positive_prob],
            title="Sentiment Confidence Scores",
            color=["Negative", "Positive"],
            color_discrete_map={"Negative": "#ff6b6b", "Positive": "#51cf66"},
            text=[f"{negative_prob:.1%}", f"{positive_prob:.1%}"]
        )
        fig.update_layout(
            showlegend=False, 
            yaxis_title="Confidence Score",
            xaxis_title="Sentiment",
            font=dict(size=12),
            height=400
        )
        fig.update_traces(textposition='outside')
        
        return result_text, fig, f"**Analysis Complete!** Sentiment: {sentiment} ({confidence:.1%} confidence)"
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", None, None

def analyze_multiple_reviews(reviews_text, max_length=128, progress=gr.Progress()):
    """
    Analyze multiple reviews from text input.
    """
    global model, tokenizer, model_loaded
    
    if not model_loaded or model is None or tokenizer is None:
        return "❌ Model not loaded. Please load the model first.", None, None
    
    if not reviews_text.strip():
        return "⚠️ Please enter reviews to analyze.", None, None
    
    try:
        # Split reviews by newlines
        reviews = [review.strip() for review in reviews_text.split('\n') if review.strip()]
        
        if not reviews:
            return "⚠️ No valid reviews found.", None, None
        
        results = []
        positive_count = 0
        negative_count = 0
        
        for i, review in enumerate(reviews):
            progress((i + 1) / len(reviews), desc=f"Analyzing review {i + 1}/{len(reviews)}...")
            
            # Tokenize and predict
            inputs = tokenizer(
                review,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            sentiment = "Positive" if predicted_class == 1 else "Negative"
            
            if sentiment == "Positive":
                positive_count += 1
            else:
                negative_count += 1
            
            results.append({
                "Review": review[:100] + "..." if len(review) > 100 else review,
                "Sentiment": sentiment,
                "Confidence": f"{confidence:.1%}"
            })
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Create summary chart
        fig = px.pie(
            values=[positive_count, negative_count],
            names=["Positive", "Negative"],
            title=f"Sentiment Distribution ({len(reviews)} reviews)",
            color_discrete_map={"Positive": "#51cf66", "Negative": "#ff6b6b"}
        )
        fig.update_layout(height=400)
        
        summary_text = f"""
### 📊 Analysis Summary

**Total Reviews Analyzed:** {len(reviews)}
**Positive Reviews:** {positive_count} ({positive_count/len(reviews)*100:.1f}%)
**Negative Reviews:** {negative_count} ({negative_count/len(reviews)*100:.1f}%)
"""
        
        return summary_text, fig, df
        
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}", None, None



def get_sample_review():
    """Return a sample restaurant review."""
    samples = [
        "The food was absolutely delicious and the service was excellent. I would definitely recommend this restaurant to anyone looking for a great dining experience!",
        "Terrible service and the food was cold when it arrived. Very disappointing experience and overpriced for what we got.",
        "Amazing atmosphere and the pasta was cooked to perfection. The staff was friendly and attentive throughout our meal.",
        "The restaurant was too noisy and the wait time was unreasonable. The food quality didn't justify the high prices.",
        "Outstanding culinary experience! Every dish was beautifully presented and bursting with flavor. Will definitely be back!"
    ]
    import random
    return random.choice(samples)

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Restaurant Review Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">🍽️ Restaurant Review Sentiment Analysis</h1>
            <p class="sub-title">AI-Powered Sentiment Classification for Restaurant Reviews</p>
        </div>
    """)
    
    # Description section
    gr.HTML("""
        <div class="description">
            <p>This application uses a <strong>fine-tuned DistilBERT model</strong> specifically trained for restaurant review sentiment analysis.
            The model <strong>RinInori/distilbert-restaurant-sentiment</strong> can classify reviews as <strong>Positive</strong> or <strong>Negative</strong> with high accuracy.</p>
        </div>
    """)
    
    with gr.Row():
        # Left column - Model Configuration
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Model Configuration")
            
            model_status = gr.Markdown(
                "Model will load automatically on startup.",
                elem_classes=["model-info"]
            )
            # Advanced settings
            with gr.Accordion("🔧 Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=128,
                    step=16,
                    label="Maximum sequence length",
                    info="Maximum length for tokenization"
                )
            
            # Model information
            gr.HTML("""
                <div class="model-info">
                    <h4>📋 Model Information</h4>
                    <p><strong>Model:</strong> <a href="https://huggingface.co/RinInori/distilbert-restaurant-sentiment" target="_blank">RinInori/distilbert-restaurant-sentiment</a></p>
                    <p><strong>Architecture:</strong> Fine-Tuned DistilBERT</p>
                    <p><strong>Training Data:</strong> European Restaurant Reviews</p>
                    <p><strong>Classes:</strong> Positive, Negative</p>
                    <p><strong>GitHub:</strong> <a href="https://github.com/hennypurwadi/FineTuning_SentimentAnalysis_Restaurant_Review/tree/main" target="_blank">View Code</a></p>
                </div>
            """)
        
        # Right column - Analysis Interface
        with gr.Column(scale=2):
            gr.Markdown("## 🔍 Sentiment Analysis")
            
            # Tab interface for different input methods
            with gr.Tabs():
                # Single Review Tab
                with gr.TabItem("Single Review"):
                    gr.Markdown("### Analyze a Single Restaurant Review")
                    
                    single_review_input = gr.Textbox(
                        label="Enter a restaurant review:",
                        placeholder="The food was absolutely delicious and the service was excellent...",
                        lines=4
                    )
                    
                    with gr.Row():
                        analyze_single_btn = gr.Button("Analyze Sentiment", variant="primary")
                        sample_btn = gr.Button("📝 Use Sample", variant="secondary")
                    
                    single_result = gr.Markdown()
                    single_chart = gr.Plot()
                    single_status = gr.Markdown()
                
                # Multiple Reviews Tab
                with gr.TabItem("Multiple Reviews"):
                    gr.Markdown("### Analyze Multiple Reviews (one per line)")
                    
                    multiple_reviews_input = gr.Textbox(
                        label="Enter multiple reviews:",
                        placeholder="The food was great!\nTerrible service, would not recommend.\nAmazing atmosphere and delicious food.",
                        lines=6
                    )
                    
                    analyze_multiple_btn = gr.Button("Analyze All Reviews", variant="primary")
                    
                    multiple_result = gr.Markdown()
                    multiple_chart = gr.Plot()
                    multiple_dataframe = gr.Dataframe()
                

    
    demo.load(load_model_and_tokenizer, outputs=model_status)
    gr.HTML("""
        <div class="description">
            <p><strong>💡 Tips:</strong> This model is specifically trained for restaurant reviews. Longer reviews may provide more accurate results. Confidence scores indicate model certainty.</p>
        </div>
    """)
    
    # Footer
    gr.HTML("""
        <div class="footer-info">
            <p>🍽️ <strong>Restaurant Review Sentiment Analysis</strong> - Powered by Fine-Tuned DistilBERT</p>
            <p><strong>Model:</strong> RinInori/distilbert-restaurant-sentiment | <strong>Built with:</strong> Transformers & Gradio</p>
        </div>
    """)
    
    # Event handlers

    
    analyze_single_btn.click(
        fn=predict_sentiment,
        inputs=[single_review_input, max_length],
        outputs=[single_result, single_chart, single_status]
    )
    
    single_review_input.submit(
        fn=predict_sentiment,
        inputs=[single_review_input, max_length],
        outputs=[single_result, single_chart, single_status]
    )
    
    sample_btn.click(
        fn=get_sample_review,
        outputs=[single_review_input]
    )
    
    analyze_multiple_btn.click(
        fn=analyze_multiple_reviews,
        inputs=[multiple_reviews_input, max_length],
        outputs=[multiple_result, multiple_chart, multiple_dataframe],
        show_progress=True
    )
    


# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )