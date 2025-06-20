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

@st.cache_resource
def load_model_and_tokenizer(model_name_or_path):
    """
    Load the fine-tuned DistilBERT model and tokenizer from Hugging Face Hub or local path.
    Uses caching to avoid reloading on every interaction.
    """
    try:
        with st.spinner(f"Loading model and tokenizer from {model_name_or_path}..."):
            # Load tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(model_name_or_path)
            
            # Load model
            model = DistilBertForSequenceClassification.from_pretrained(model_name_or_path)
            
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
    
    st.markdown("""
    This application uses a fine-tuned **DistilBERT** model to analyze the sentiment of restaurant reviews.
    The model was trained on European Restaurant Reviews dataset and can classify reviews as **Positive** or **Negative**.
    """)
    
    # Sidebar for model configuration
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Model Configuration</h2>', unsafe_allow_html=True)
    
    # Model source selection
    model_source = st.sidebar.radio(
        "Choose model source:",
        ["Hugging Face Hub", "Local Path"],
        help="Select whether to load from Hugging Face Hub or a local directory"
    )
    
    if model_source == "Hugging Face Hub":
        model_name = st.sidebar.text_input(
            "Model name on Hugging Face Hub:",
            value="distilbert-base-uncased",
            help="Enter the model name from Hugging Face Hub (e.g., 'username/model-name')"
        )
    else:
        model_name = st.sidebar.text_input(
            "Local model path:",
            value="/path/to/your/fine-tuned-model",
            help="Enter the full path to your local model directory"
        )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        max_length = st.slider(
            "Maximum sequence length:",
            min_value=64,
            max_value=512,
            value=128,
            help="Maximum length for tokenization"
        )
        
        batch_processing = st.checkbox(
            "Enable batch processing",
            help="Process multiple reviews at once"
        )
    
    # Load model button
    if st.sidebar.button("🚀 Load Model", type="primary"):
        model, tokenizer = load_model_and_tokenizer(model_name)
        if model and tokenizer:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
    
    # Check if model is loaded
    if not hasattr(st.session_state, 'model_loaded') or not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar.")
        
        # Show example of how to use with a pre-trained model
        st.markdown("### 📝 Getting Started")
        st.markdown("""
        1. **For Hugging Face Hub**: Enter a model name like `distilbert-base-uncased` or your fine-tuned model
        2. **For Local Path**: Enter the full path to your saved model directory
        3. Click **Load Model** to initialize the model
        4. Start analyzing restaurant reviews!
        
        **Example model names:**
        - `distilbert-base-uncased` (base model)
        - `your-username/fine-tuned-restaurant-sentiment` (your fine-tuned model)
        """)
        return
    
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
            st.info(f"📍 Model Source: {model_name}")
            
            # Model details
            with st.expander("🔍 Model Details"):
                st.write("**Architecture:** DistilBERT")
                st.write("**Task:** Sentiment Classification")
                st.write("**Classes:** Positive, Negative")
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
                st.session_state.sample_text = sample
        
        # Instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Load Model**: Use the sidebar to load your model
            2. **Choose Input**: Select single review, multiple reviews, or file upload
            3. **Analyze**: Click the analyze button to get sentiment predictions
            4. **Review Results**: See sentiment classification and confidence scores
            
            **Tips:**
            - Longer reviews may provide more accurate results
            - The model works best with restaurant-related content
            - Confidence scores indicate model certainty
            """)

if __name__ == "__main__":
    main()

