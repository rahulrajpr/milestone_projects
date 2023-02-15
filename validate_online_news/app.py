import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

import predictor

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('interface.html')


@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    pred_value = predictor.prediction(news)

    ini_text = 'The news you read online is probably a '
    if pred_value == 0:
        display_value = 'Fake News'
    else:
        display_value = 'Valid News'

    return render_template('interface.html', prediction_text=ini_text + display_value)


if __name__ == "__main__":
    app.run(debug=True)
