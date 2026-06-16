from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
import io
import base64
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

app = Flask(__name__)

# Load model globally at startup to optimize performance (avoid reloading on every request)
MODEL_PATH = 'cancer.h5'
logger.info(f"Loading Keras model from {MODEL_PATH}...")
model = load_model(MODEL_PATH, compile=False)
logger.info("Model loaded successfully.")
logger.info(f"Model expected input shape: {model.input_shape}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/show', methods=['POST', 'GET'])
def show():
    if request.method == "POST":
        img_file = request.files["image"]
        img = Image.open(img_file)
        
        # Save image to bytes to render on output page
        data = io.BytesIO()
        img.save(data, "JPEG")
        img_filepath = base64.b64encode(data.getvalue())
        
        # Run prediction
        prediction = output(data)
        
        return render_template('output.html', image=img_filepath.decode('utf-8'), predict=prediction)

def output(img_bytes_io):
    # Reset bytes stream to the beginning
    img_bytes_io.seek(0)
    
    # Load image in RGB format with target shape (224, 224)
    img = load_img(img_bytes_io, target_size=(224, 224), color_mode='rgb')
    
    # Convert image to numpy array
    x = img_to_array(img)
    image_shape = x.shape
    min_pixel = np.min(x)
    max_pixel = np.max(x)
    
    # Add batch dimension: shape becomes (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    tensor_shape = x.shape
    
    # Perform inference (DO NOT scale x by dividing by 255.0)
    pred = model.predict(x)
    raw_pred_val = float(pred[0][0])
    
    # Log debugging details
    logger.info("========== INFERENCE DEBUGGING LOGS ==========")
    logger.info(f"Loaded image shape: {image_shape}")
    logger.info(f"Input tensor shape: {tensor_shape}")
    logger.info(f"Pixel values range: Min = {min_pixel:.1f}, Max = {max_pixel:.1f}")
    logger.info(f"Raw prediction value (before thresholding): {raw_pred_val:.8f}")
    logger.info("==============================================")
    
    # Apply thresholding of 0.5 for binary sigmoid activation
    if raw_pred_val < 0.5:
        return "You are completely normal and not affected with cancer."
    else:
        return "High risk!!! Cancer cells spots detected in your pathological image."

if __name__ == '__main__':
    app.run(debug=True)