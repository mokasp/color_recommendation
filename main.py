#!/usr/bin/env python3
from flask import Flask, render_template, url_for, request, jsonify
import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2


app = Flask(__name__)
model = tf.keras.models.load_model('test_model_00.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        data_url = request.form['image']  # base64 string

        if data_url.startswith('data:image'):
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')


                return f'<h2>Image received!</h2><img src="data:image/jpeg;base64,{img_base64}" width="1280">'
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('process_image.html')
