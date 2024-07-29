from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained content-based model and user similarity matrix
content_model = joblib.load('content_model.pkl')
user_similarity = np.load('user_similarity.npy')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve inputs from the form
        input_data = {
            'Daily Time Spent on Site': float(request.form['time_spent']),
            'Age': int(request.form['age']),
            'Area Income': float(request.form['income']),
            'Daily Internet Usage': float(request.form['internet_usage']),
            'Male': int(request.form['male'])
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Preprocess input data
        X_input = content_model.named_steps['preprocessor'].transform(df)

        # Use the trained model to make predictions
        prediction = content_model.named_steps['classifier'].predict(X_input)

        # Generate recommendations based on predictions
        recommendation = 'Ad will be clicked' if prediction[0] == 1 else 'Ad will not be clicked'

        return render_template('result.html', recommendation=recommendation)
    except KeyError as e:
        return f"KeyError: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
