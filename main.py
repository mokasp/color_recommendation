#!/usr/bin/env python3
from flask import Flask
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('test_model_00.keras')

@app.route('/')
def hello_world():
    return 'Hello, World!'