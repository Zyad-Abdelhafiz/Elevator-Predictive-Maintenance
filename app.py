from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import pickle
import pandas as pd
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

pkl_filename = "my_model.pkl"
with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

 

def predict_mpg(data):
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    
    
    # Convert DataFrame to NumPy array and reshape
    input_data = df.values.reshape((1, 50, 3)) 
    
    # Get prediction probabilities
    y_pred_prob = model.predict(input_data)
    
    # Apply threshold to get binary predictions
    y_pred = (y_pred_prob > 0.5).astype("int32")
    
    # Interpret the prediction result
    if y_pred[0] == 0:
        return 'No Need Maintenance'
    elif y_pred[0] == 1:
        return 'Need Maintenance'
    else:
        return 'Unknown prediction result'


class Test(Resource):
    def get(self):
        return 'Welcome to Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if value:
                return {'Post Values': value}, 201
            return {"error": "Invalid format."}
        except Exception as error:
            return {'error': str(error)}

class GetPredictionOutput(Resource):
    def get(self):
        return {"error": "Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            # Ensure the input data is in the correct format
            predict_output = predict_mpg(data)
            return {'predict': predict_output}
        except Exception as error:
            return {'error': str(error)}

api.add_resource(Test, '/')
api.add_resource(GetPredictionOutput, '/getPredictionOutput')

if __name__ == '__main__':
    app.run(debug=True)
