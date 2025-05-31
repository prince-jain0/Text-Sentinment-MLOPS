import streamlit as st
import pandas as pd
import pickle
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from helper_functions import log_info, log_error

# Load environment variables
load_dotenv()

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_classifier.pkl")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

# Load model artifacts
def load_artifact(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        log_error(f"Artifact not found: {filepath}")
        st.error(f"Error: Artifact not found: {filepath}")
        return None

def predict_sentiment(text):
    pipeline = load_artifact(PIPELINE_PATH)
    model = load_artifact(MODEL_PATH)
    label_encoder = load_artifact(LABEL_ENCODER_PATH)

    if not pipeline or not model or not label_encoder:
        return None

    transformed_input = pipeline.transform(pd.Series([text]))
    prediction = model.predict(transformed_input)
    return label_encoder.inverse_transform(prediction)[0]

# Streamlit UI setup
st.set_page_config(page_title="Web & Manual Sentiment Analyzer", layout="wide")
st.title("💬 Web + Manual Text Sentiment Analyzer")

st.markdown("### Option 1: 🌐 Scrape Content from a Webpage")

# URL Input
url = st.text_input("Enter a webpage URL:", placeholder="https://en.wikipedia.org/wiki/Sentiment_analysis")

if st.button("Fetch Web Content"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract readable paragraph content
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            full_text = "\n\n".join(paragraphs)

            if not full_text:
                st.warning("No visible text content found on the page.")
            else:
                st.session_state["scraped_text"] = full_text[:15000]  # Limit text for performance
                with st.expander("📄 Full Webpage Content (Click to Expand)", expanded=False):
                    st.code(full_text, language="markdown")

                # Group and display paragraphs
                st.markdown("### 📚 Grouped Paragraph Sections")

                GROUP_SIZE = 3
                MAX_SECTIONS = 10

                grouped_paragraphs = [
                    "\n\n".join(paragraphs[i:i + GROUP_SIZE])
                    for i in range(0, len(paragraphs), GROUP_SIZE)
                ]

                for i, group in enumerate(grouped_paragraphs[:MAX_SECTIONS]):
                    with st.expander(f"📝 Section {i + 1}"):
                        st.write(group)

                if len(grouped_paragraphs) > MAX_SECTIONS:
                    if st.checkbox("Show more sections"):
                        for i, group in enumerate(grouped_paragraphs[MAX_SECTIONS:], start=MAX_SECTIONS + 1):
                            with st.expander(f"📝 Section {i}"):
                                st.write(group)
        except Exception as e:
            st.error(f"Failed to fetch page: {e}")

# Paste or Select Text from Web Content
if "scraped_text" in st.session_state:
    st.markdown("### ✂️ Copy text from above and paste below to analyze sentiment:")
    user_web_text = st.text_area("Paste selected paragraph or text", height=150, key="web_input")
    if st.button("Analyze Web Text Sentiment"):
        if user_web_text.strip():
            sentiment = predict_sentiment(user_web_text)
            st.success(f"🧠 Sentiment: **{sentiment}**")
            log_info(f"Web Sentiment Prediction: {sentiment}")
        else:
            st.warning("Please paste some text to analyze.")

# Manual Input
st.markdown("---")
st.markdown("### ✍️ Option 2: Manually Type Text")
manual_text = st.text_area("Type or paste text here for sentiment analysis:", height=150, key="manual_input")

if st.button("Analyze Manual Text Sentiment"):
    if manual_text.strip():
        sentiment = predict_sentiment(manual_text)
        st.success(f"🧠 Sentiment: **{sentiment}**")
        log_info(f"Manual Sentiment Prediction: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
