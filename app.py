from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the model
model = tf.keras.models.load_model('my_model.keras')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert JSON data to DataFrame or the format required by your model
        df = pd.DataFrame([data])

        # Make predictions
        predictions = model.predict(df)
        
        # Convert predictions to a list (or format as needed)
        response = predictions.tolist()
        
        # Return the predictions as JSON
        return jsonify({'predictions': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
