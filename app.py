from flask import Flask, render_template, request, jsonify
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
import pandas as pd
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    if request.method == 'POST':
        try:
            applicant_income = float(request.form['applicant_income'])
            coapplicant_income = float(request.form['coapplicant_income'])
            loan_amount = float(request.form['loan_amount'])
            loan_amount_term = float(request.form['loan_amount_term'])
            credit_history = float(request.form['credit_history'])
            gender = request.form['gender']
            married = request.form['married']
            dependents = request.form['dependents']
            education = request.form['education']
            self_employed = request.form['self_employed']
            property_area = request.form['property_area']

            # Create CustomData instance
            custom_data = CustomData(
                applicant_income=applicant_income,
                coapplicant_income=coapplicant_income,
                loan_amount=loan_amount,
                loan_amount_term=loan_amount_term,
                credit_history=credit_history,
                gender=gender,
                married=married,
                dependents=dependents,
                education=education,
                self_employed=self_employed,
                property_area=property_area
            )

            data_df = custom_data.get_data_as_dataframe()

            prediction_pipeline = PredictPipeline()
            predictions = prediction_pipeline.predict(features=data_df)

            print(f"\nRaw Prediction: {predictions}")

            if predictions[0] == 'Y':
                result = "Loan Application Approved"
            else:
                result = "Loan Application Denied"

            return render_template('predict.html', prediction=result)

        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('error.html', message="There was an error with the prediction. Please try again.")
    
    return render_template('get_data.html')

@app.route('/train_model', methods=['GET'])
def train_model():
    try:
        subprocess.Popen(['python3', '-m', 'src.pipelines.training_pipeline'])

        return jsonify({"message": "Training has started. Please check the logs for progress."})

    except Exception as e:
        print(f"Error occurred while training the model: {e}")
        return jsonify({"message": f"An error occurred while training the model: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
