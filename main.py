from flask import Flask, request, render_template
import pickle

import numpy as np

app = Flask(__name__)

# load model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # inp data
    time_of_study = request.form['ToS']

    # prediction by model
    input_data = [[float(time_of_study)]]

    reshaped_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(reshaped_data)

    # value transfer

    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
