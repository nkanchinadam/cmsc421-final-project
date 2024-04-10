from flask import Flask, request, render_template
import tensorflow as tf
# from model import base_model, IMG_SIZE  # Import your model and any necessary variables
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = np.asarray(image)
    image = np.expand_dims(image, 0)
    return image

@app.route('/', methods=['GET'])
def index():
    # Serve the upload form on GET request
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        image = Image.open(io.BytesIO(file.read()))
        image = prepare_image(image, IMG_SIZE)
        
        # Make sure your model is in prediction mode
      #  base_model.trainable = False
        
        # Predict
        predictions = base_model.predict(image)
        
        # Process your predictions here
        # For example, you might want to return the top prediction
        top_prediction = predictions[0]
        
        # Return the result as a string (customize as needed)
        return str(top_prediction)

if __name__ == '__main__':
    app.run(debug=True)