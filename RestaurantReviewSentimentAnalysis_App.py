import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import pandas as pd
import plotly.express as px
import time

# Configure page
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Default model configuration
DEFAULT_MODEL = "RinInori/distilbert-restaurant-sentiment"

@st.cache_resource
def load_model_and_tokenizer(model_name=DEFAULT_MODEL):
    """
    Load the fine-tuned DistilBERT model and tokenizer from Hugging Face Hub.
    Uses caching to avoid reloading on every interaction.
    """
    try:
        with st.spinner(f"Loading model and tokenizer from {model_name}..."):
            # Load tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            
            # Load model
            model = DistilBertForSequenceClassification.from_pretrained(model_name)
            
            # Set model to evaluation mode
            model.eval()
            
        st.success("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer, max_length=128):
    """
    Predict sentiment for a given text using the loaded model.
    """
    if not text.strip():
        return None, None
    
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
        
        # Map class to sentiment (0: Negative, 1: Positive)
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        
        return sentiment, confidence
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🍽️ Restaurant Review Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    This application uses a **fine-tuned DistilBERT model** specifically trained for restaurant review sentiment analysis.
    The model **{DEFAULT_MODEL}** can classify reviews as **Positive** or **Negative** with high accuracy.
    """)
    
    # Sidebar for model configuration
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Model Configuration</h2>', unsafe_allow_html=True)
    
    # Model information
    st.sidebar.info(f"**Model:** {DEFAULT_MODEL}")
    st.sidebar.markdown("*Fine-tuned on European Restaurant Reviews*")
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        max_length = st.slider(
            "Maximum sequence length:",
            min_value=64,
            max_value=512,
            value=128,
            help="Maximum length for tokenization"
        )
        
        # Option to use a different model
        use_custom_model = st.checkbox("Use different model", help="Load a different Hugging Face model")
        
        if use_custom_model:
            custom_model = st.text_input(
                "Custom model name:",
                value=DEFAULT_MODEL,
                help="Enter any Hugging Face model name"
            )
            model_to_load = custom_model
        else:
            model_to_load = DEFAULT_MODEL
    
    # Auto-load the default model on startup
    if 'model_loaded' not in st.session_state:
        model, tokenizer = load_model_and_tokenizer(model_to_load)
        if model and tokenizer:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.session_state.current_model = model_to_load
    
    # Load model button (for custom models)
    if use_custom_model and st.sidebar.button("🚀 Load Custom Model", type="primary"):
        model, tokenizer = load_model_and_tokenizer(model_to_load)
        if model and tokenizer:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.session_state.current_model = model_to_load
    
    # Check if model is loaded
    if not hasattr(st.session_state, 'model_loaded') or not st.session_state.model_loaded:
        st.warning("⚠️ Model is loading, please wait...")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Analyze Restaurant Review</h2>', unsafe_allow_html=True)
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Single Review", "Multiple Reviews", "Upload File"],
            horizontal=True
        )
        
        if input_method == "Single Review":
            # Single text input
            review_text = st.text_area(
                "Enter a restaurant review:",
                height=150,
                placeholder="The food was absolutely delicious and the service was excellent. I would definitely recommend this restaurant to anyone looking for a great dining experience!"
            )
            
            if st.button("🔍 Analyze Sentiment", type="primary"):
                if review_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        sentiment, confidence = predict_sentiment(
                            review_text, 
                            st.session_state.model, 
                            st.session_state.tokenizer,
                            max_length
                        )
                    
                    if sentiment:
                        # Display results
                        if sentiment == "Positive":
                            st.markdown(f"""
                            <div class="sentiment-positive">
                                <h3>😊 Positive Sentiment</h3>
                                <p class="confidence-score">Confidence: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="sentiment-negative">
                                <h3>😞 Negative Sentiment</h3>
                                <p class="confidence-score">Confidence: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        fig = px.bar(
                            x=["Negative", "Positive"],
                            y=[1-confidence if sentiment == "Positive" else confidence, 
                               confidence if sentiment == "Positive" else 1-confidence],
                            title="Sentiment Confidence Scores",
                            color=["Negative", "Positive"],
                            color_discrete_map={"Negative": "#ff6b6b", "Positive": "#51cf66"}
                        )
                        fig.update_layout(showlegend=False, yaxis_title="Confidence Score")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please enter a review to analyze.")
        
        elif input_method == "Multiple Reviews":
            # Multiple reviews input
            st.markdown("Enter multiple reviews (one per line):")
            multiple_reviews = st.text_area(
                "Reviews:",
                height=200,
                placeholder="The food was great!\nTerrible service, would not recommend.\nAmazing atmosphere and delicious food."
            )
            
            if st.button("🔍 Analyze All Reviews", type="primary"):
                if multiple_reviews.strip():
                    reviews = [review.strip() for review in multiple_reviews.split('\n') if review.strip()]
                    
                    if reviews:
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, review in enumerate(reviews):
                            sentiment, confidence = predict_sentiment(
                                review, 
                                st.session_state.model, 
                                st.session_state.tokenizer,
                                max_length
                            )
                            results.append({
                                "Review": review[:100] + "..." if len(review) > 100 else review,
                                "Sentiment": sentiment,
                                "Confidence": f"{confidence:.2%}" if confidence else "N/A"
                            })
                            progress_bar.progress((i + 1) / len(reviews))
                        
                        # Display results table
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        positive_count = sum(1 for r in results if r["Sentiment"] == "Positive")
                        negative_count = len(results) - positive_count
                        
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.metric("Positive Reviews", positive_count)
                        with col_neg:
                            st.metric("Negative Reviews", negative_count)
                else:
                    st.warning("Please enter reviews to analyze.")
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload a CSV file with reviews",
                type=['csv'],
                help="CSV should have a column named 'review' or 'Review'"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Find review column
                    review_columns = [col for col in df.columns if 'review' in col.lower()]
                    if review_columns:
                        review_col = st.selectbox("Select review column:", review_columns)
                        
                        if st.button("🔍 Analyze Uploaded Reviews", type="primary"):
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, review in enumerate(df[review_col].dropna()):
                                sentiment, confidence = predict_sentiment(
                                    str(review), 
                                    st.session_state.model, 
                                    st.session_state.tokenizer,
                                    max_length
                                )
                                results.append({
                                    "Review": str(review)[:100] + "..." if len(str(review)) > 100 else str(review),
                                    "Sentiment": sentiment,
                                    "Confidence": f"{confidence:.2%}" if confidence else "N/A"
                                })
                                progress_bar.progress((i + 1) / len(df[review_col].dropna()))
                            
                            # Display results
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "sentiment_analysis_results.csv",
                                "text/csv"
                            )
                    else:
                        st.error("No review column found. Please ensure your CSV has a column named 'review' or 'Review'.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.markdown('<h2 class="sub-header">ℹ️ Model Information</h2>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
            st.success("✅ Model Status: Loaded")
            st.info(f"📍 Current Model: {st.session_state.current_model}")
            
            # Model details
            with st.expander("🔍 Model Details"):
                st.write("**Architecture:** DistilBERT")
                st.write("**Task:** Restaurant Sentiment Classification")
                st.write("**Classes:** Positive, Negative")
                st.write("**Training:** European Restaurant Reviews")
                st.write(f"**Max Length:** {max_length} tokens")
        
        # Sample reviews for testing
        st.markdown('<h3 class="sub-header">📋 Sample Reviews</h3>', unsafe_allow_html=True)
        
        sample_reviews = [
            "The food was absolutely delicious and the service was outstanding!",
            "Terrible experience. The food was cold and the staff was rude.",
            "Average restaurant with decent food but nothing special.",
            "Best dining experience I've had in years! Highly recommended.",
            "Overpriced and underwhelming. Would not return."
        ]
        
        for i, sample in enumerate(sample_reviews):
            if st.button(f"Try Sample {i+1}", key=f"sample_{i}"):
                # Use session state to pass the sample text
                st.session_state.sample_text = sample
                st.rerun()
        
        # Apply sample text if available
        if 'sample_text' in st.session_state:
            # This will be handled by the text area value
            pass
        
        # Instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Model Ready**: The fine-tuned model is automatically loaded
            2. **Choose Input**: Select single review, multiple reviews, or file upload
            3. **Analyze**: Enter text and click the analyze button
            4. **Review Results**: See sentiment classification and confidence scores
            
            **Tips:**
            - This model is specifically trained for restaurant reviews
            - Longer reviews may provide more accurate results
            - Confidence scores indicate model certainty
            - Use the sample reviews to test the functionality
            """)

if __name__ == "__main__":
    main()

