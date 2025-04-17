#!/usr/bin/env python3
from flask import Flask, render_template, url_for
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('test_model_00.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')