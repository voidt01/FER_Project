import cv2
import os
import joblib
import logging
import gradio as gr
from src.preprocessing import preprocess
from src.featureExtraction import extract_with_hog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(model_path=os.path.join('model', 'svmC_v1.pkl')):
    """Loaded the trained SVM model"""
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        logging.info(f'Error msg: {e}')
        return None

def preprocess_predict_image(image):
    """Process given image (by webcam or uploaded) and make a prediction from there """
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        model_face_cascade_path = os.path.join('model', 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(model_face_cascade_path)

        detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(detected_face) < 1:
            logging.info('There is no face detected')
            return new_image, 'No Face Detected' 
        
        x, y, w, h = detected_face[0]
        face_img = new_image[y:y+h, x:x+w]

        cv2.rectangle(new_image, (x, y), (x+w, y+h), (0, 255, 0), 1)

        preprocessed_face = preprocess(face_img)
        face_features = extract_with_hog(preprocessed_face)

        class_list = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Suprise']

        model = load_model()
        if model is None:
            return new_image, "Error loading model"
        
        prediction = model.predict([face_features])[0]
        
        return new_image, f"Prediction: {class_list[prediction]}"

    except Exception as e:
        logging.info(f'Error msg: {e}')
        return image, f"Error: {e}"
    

def create_gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Face Detection and Classification")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(sources=["upload"], type="numpy", label="Upload Image")
                webcam_input = gr.Image(sources=["webcam"], type="numpy", label="Or Use Webcam")
                
                
                with gr.Row():
                    upload_button = gr.Button("Process Uploaded Image", size="lg")
                    webcam_button = gr.Button("Process Webcam Image", size="lg")
            
            
            with gr.Column(scale=1):  
                with gr.Group():  
                    output_image = gr.Image(label="Processed Image")
                    output_text = gr.Textbox(
                        label="Prediction Result",
                        show_label=True,
                        container=True,
                        scale=1
                    )
    
        upload_button.click(
            fn=preprocess_predict_image,
            inputs=image_input,
            outputs=[output_image, output_text]
        )
        webcam_button.click(
            fn=preprocess_predict_image,
            inputs=webcam_input,
            outputs=[output_image, output_text]
        )
    
    return app

if __name__ == '__main__':
    app = create_gradio_interface()
    app.launch(share=True)