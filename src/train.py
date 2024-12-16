import joblib
import os
import numpy as np
import logging 

from dataLoader import LoadData
from preprocessing import preprocess
from featureExtraction import extract_with_hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def train_model(X_train, y_train):
    """
    Train a SVM classifier model

    Args:
        X_train (np.ndarray): Features for training
        y_train (np.ndarray): labels for training
    
    Returns:
        sklearn.svm.SVC: Trained SVM model

    """
    logging.info("Training model has started")
    model = SVC(kernel='rbf', gamma='scale', C=1, probability=True) 
    model.fit(X_train, y_train) 
    logging.info("Model training completed")
    return model

def save_model(model, filename):
    logging.info(f"saving model to {filename}")
    joblib.dump(model, filename) 
    logging.info("Model saved successfully")

def main():
    try:
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(ROOT_DIR, 'data')
        model_path = os.path.join(ROOT_DIR, 'model', 'svmC_v1.pkl')

        logging.info("Starting training process.")
        img_dataset = LoadData(data_path)

        features = np.array([extract_with_hog(preprocess(img)) for img, _ in img_dataset])
        labels = np.array(img_dataset.image_labels)

        X_train, _, y_train, _ = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

        model = train_model(X_train, y_train)

        # Ensure model directory exists
        os.makedirs(os.path.join(ROOT_DIR, 'model'), exist_ok=True)

        # Save the model
        logging.info(f"Saving model to: {model_path}")
        save_model(model, model_path)
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == '__main__':
    main()
