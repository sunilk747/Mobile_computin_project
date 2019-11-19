import pickle
import operator

import pandas as pd

predicted_values = {
    0: "buy",
    1: "communicate",
    2: "fun",
    3: "hope",
    4: "mother",
    5: "really"
}

def process_output(p_values):
    res = dict()
    for item in p_values:
        if res.get(item):
            res[item] += 1
        else:
            res[item] = 1
    val = max(res.items(), key=operator.itemgetter(1))[0]
    return predicted_values[val]

def process_input(data):
    final_data = list()
    for frame in data:
        points = dict()
        for item in frame['keypoints']:
            key1 = item['part'] + '_score'
            key2 = item['part'] + '_x'
            key3 = item['part'] + '_y'
            points[key1] = item['score']
            points[key2] = item['position']['x']
            points[key3] = item['position']['y']
        final_data.append(points)
    input_data = pd.DataFrame(final_data)
    return input_data


class Models:

    def logistic_regression(self, input):
        model_input = process_input(input)
        lr_model = pickle.load(open('models/logistic_regression.sav', 'rb'))
        lr_predictions = lr_model.predict(model_input)
        output = process_output(lr_predictions)
        print (output)
        return output

    def k_neighbour(self, input):
        model_input = process_input(input)
        kn_model = pickle.load(open('models/k_neighbour.sav', 'rb'))
        kn_predictions = kn_model.predict(model_input)
        output = process_output(kn_predictions)
        print (output)
        return output

