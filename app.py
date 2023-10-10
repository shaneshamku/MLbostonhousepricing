import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas
import os


app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set


## Load Model
regressionModel = pickle.load(open('regmodel.pkl','rb'))
scalarModel = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalarModel.transform(np.array(list(data.values())).reshape(1,-1))
    output = regressionModel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalarModel.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regressionModel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The House price prediction is {}.".format(output))

if __name__=="__main__":
    app.run(host='0.0.0.0', port=port)
