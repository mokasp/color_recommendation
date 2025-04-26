#!/usr/bin/env python3
from flask import Flask, render_template, url_for, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from app.colorspace import load_colors, process_color_lists
from app.predictions import predict
import mediapipe as mp
import logging


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)



colors, color_lists = load_colors()
color_lists_lab = process_color_lists(color_lists)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/button1')
def button1():
    return render_template('button1.html')

@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        data_url = request.form['image']  # base64 string

        if data_url.startswith('data:image'):
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (960, 540))
            logging.debug(f"image: {np_arr}")

            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                # load MediaPipe Face Mesh
                mp_face_mesh = mp.solutions.face_mesh

                prediction, input_vectors_lab, output_lab = predict(resized_img, mp_face_mesh)
                logging.debug(f"prediction: {prediction}")
                # logging.debug(f"input_lab_vectors: {input_vectors_lab}")
                # logging.debug(f"output_lab: {output_lab}")


                return f'<h2>Image received!</h2><img src="data:image/jpeg;base64,{img_base64}" width="1280">'
                # return 'prediction'
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('process_image.html')
