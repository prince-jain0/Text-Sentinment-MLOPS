import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from helper_functions import log_info, log_error
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


def training_pipeline(X_train, y_train):
    """
    Trains a Logistic Regression classifier and logs with MLflow.
    """
    try:
        mlflow.set_experiment("Sentiment-Analysis")
        with mlflow.start_run():
            model = LogisticRegression(max_iter=1000, n_jobs=-1)
            model.fit(X_train, y_train)

            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)
            mlflow.sklearn.log_model(model, "model")

            log_info(f"✅ Logistic Regression model trained and saved at {MODEL_PATH}")
            return model
    except Exception as e:
        log_error(f"❌ Error during Logistic Regression model training: {e}")
        raise
def load_model():
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

def prediction_pipeline(X_val):
    model = load_model()
    with open(LABEL_ENCODER_PATH, 'rb') as file:
        label_encoder = pickle.load(file)
    predictions = model.predict(X_val)
    return label_encoder.inverse_transform(predictions)

def evaluation_matrices(X_val, y_val):
    preds = prediction_pipeline(X_val)
    with open(LABEL_ENCODER_PATH, 'rb') as file:
        label_encoder = pickle.load(file)
    y_true = label_encoder.inverse_transform(y_val)
    cm = confusion_matrix(y_true, preds, labels=label_encoder.classes_)
    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds)
    log_info("Model evaluation completed.")
    log_info(f"Confusion Matrix:\n{cm}")
    log_info(f"Accuracy Score: {acc}")
    log_info(f"Classification Report:\n{report}")
    return cm, acc, report
