from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import json

def predict_epm(config):
    ##loading the model from the saved file
    filename = "my_model.keras"
    with open(filename, 'rb') as f_in:
        model = tf.keras.models.load_model(f_in)

    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    y_pred = model.predict(df)
    
    if y_pred == 0:
        return 'No Need Maintenance'
    elif y_pred == 1:
        return 'Need Maintenance'
