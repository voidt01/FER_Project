import joblib 
from preprocessing import preprocess
from featureExtraction import extract_with_hog

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict(model, img):
    features = extract_with_hog(preprocess(img))
    prediction = model.predict(features)
    return prediction

