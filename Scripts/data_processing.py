import os
import pickle
import re
import string
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from helper_functions import log_info

nltk.download('stopwords')
from nltk.corpus import stopwords

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def create_data_pipeline():
    vectorizer = CountVectorizer()
    pipeline = Pipeline([("vectorizer", vectorizer)])
    log_info("Text vectorization pipeline created.")
    return pipeline

def save_pipeline(pipeline):
    with open(PIPELINE_PATH, 'wb') as file:
        pickle.dump(pipeline, file)
    log_info(f"Pipeline saved at {PIPELINE_PATH}")

def encode_response_variable(y):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
    return y_encoded
