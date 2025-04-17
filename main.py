#!/usr/bin/env python3
from flask import Flask, render_template, url_for, request, jsonify
import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO
import numpy as np


app = Flask(__name__)
model = tf.keras.models.load_model('test_model_00.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/process_image', methods=['POST', 'GET'])
def process_image():
    data = request.get_json()
    image_data = data['image']

    encoded = image_data.split(",")[1]
    decoded = base64.b64decode(encoded)
    image = Image.open(BytesIO(decoded)).convert('RGB')
    image_np = np.array(image)

    return jsonify({"status": "ok", "message": "Image recieved and processed"})