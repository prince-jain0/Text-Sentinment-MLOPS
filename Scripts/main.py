import os
import pandas as pd
from dotenv import load_dotenv
from data_processing import create_data_pipeline, save_pipeline, encode_response_variable
from ml_functions import training_pipeline, evaluation_matrices
from helper_functions import log_info

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, os.getenv("DATA_DIR"), "raw", "train.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv("ARTIFACTS_DIR"))
print(DATA_PATH)
def main():
    log_info("🚀 Starting Sentiment Analysis Pipeline")
    df = pd.read_csv(DATA_PATH, encoding='latin1')
    df.dropna(inplace=True)
    df['text'] = df['text'].astype(str)
    
    from data_processing import clean_text, remove_stopwords
    df['cleaned_text'] = df['text'].apply(clean_text).apply(remove_stopwords)

    X = df['cleaned_text']
    y = df['sentiment']
    y_encoded = encode_response_variable(y)

    pipeline = create_data_pipeline()
    X_transformed = pipeline.fit_transform(X)
    save_pipeline(pipeline)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)

    training_pipeline(X_train, y_train)
    evaluation_matrices(X_val, y_val)

if __name__ == "__main__":
    main()
