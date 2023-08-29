# # app.py (Flask Backend)
# from flask import Flask, render_template, request, jsonify
# import pickle
# import pandas as pd
# import numpy as np
# import warnings
#
# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=Warning)
#
# app = Flask(__name__, template_folder='template')
#
# model = pickle.load(open('C:/Users/faiza/Downloads/XGB-1.pkl', 'rb'))
#
# @app.route('/')
# def index():
#     return render_template('fff.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     if file:
#         df = pd.read_csv(file)
#         x_values = df[['AccV', 'AccML', 'AccAP']].values  # Replace with actual X attribute names
#
#         predicted_values = model.predict(x_values)
#
#         chart_data = {
#             'labels': list(range(1, len(predicted_values) + 1)),
#             'predicted_values': predicted_values.tolist(),
#             'x_values': {
#                 'AccV': df['AccV'].tolist(),
#                 'AccML': df['AccML'].tolist(),
#                 'AccAP': df['AccAP'].tolist()
#             }
#         }
#
#         return jsonify(chart_data)
#     return jsonify({'error': 'No file uploaded'})
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import warnings
# from gunicorn.app.base import Application
# from gunicorn import util
# import fcntl

warnings.filterwarnings("ignore", category=Warning)

app = Flask(__name__, template_folder='template')

home_model = pickle.load(open('C:/Users/faiza/Downloads/dt_unbalanced.pkl', 'rb'))
lab_model = pickle.load(open('C:/Users/faiza/Downloads/LGBM_IMBalanced_LabData.pkl2', 'rb'))


@app.route('/')
def index():
    return render_template('fff.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        x_values = df[['AccV', 'AccML', 'AccAP']].values

        selected_model = request.form.get('model')
        if selected_model == 'home':
            model = home_model
        else:
            model = lab_model

        predicted_values = model.predict(x_values)

        chart_data = {
            'labels': list(range(1, len(predicted_values) + 1)),
            'predicted_values': predicted_values.tolist(),
            'x_values': {
                'AccV': df['AccV'].tolist(),
                'AccML': df['AccML'].tolist(),
                'AccAP': df['AccAP'].tolist()
            }
        }

        return jsonify(chart_data)
    return jsonify({'error': 'No file uploaded'})


if __name__ == '__main__':
    app.run(debug=True)
