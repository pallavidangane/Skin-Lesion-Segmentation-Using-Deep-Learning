from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
 
app = Flask(__name__)
 
# Load your pre-trained model
model = tf.keras.models.load_model('C:/Users/PALLAVI/OneDrive/Desktop/my_project/model/unet_100_epoch.h5')
 
# Function to preprocess the input image for the UNet model
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Resize to model's expected input size                  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
 
# Function to postprocess the model's output and convert it to base64 for display
def postprocess_image(segmented):
    segmented = np.squeeze(segmented)  # Remove batch dimension
    segmented = (segmented * 255).astype(np.uint8)  # Convert to grayscale
    image_pil = Image.fromarray(segmented)
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/segment', methods=['POST'])
def segment_image():
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    img = preprocess_image(image)
    pred = model.predict(img)
    pred = (pred > 0.5).astype(np.uint8)  # Binary mask
    segmented_img_str = postprocess_image(pred[0])
    return jsonify({
        'segmented_image': segmented_img_str
    })
 
if __name__ == '__main__':
    app.run(debug=True)