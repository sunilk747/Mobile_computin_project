from flask import Flask, request
from flask.json import jsonify

from models import Models

application = Flask(__name__)

@application.route('/')
def status():
    return jsonify({"Status":"OK"})


@application.route('/predict', methods=['POST'])
def predict():
    output = {}
    input_json = request.get_json(force=True)
    model = Models()
    model1 = model.k_neighbour(input_json)
    model2 = model.logistic_regression(input_json)
    output['K_Neighbour'] = model1
    output['Logistic_Regression'] = model2
    return jsonify(output), 200



if __name__ =='__main__':
    application.debug = True
    application.run(host='0.0.0.0')

