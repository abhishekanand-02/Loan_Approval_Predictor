import os
import sys
import pickle
import mlflow
import mlflow.sklearn
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from src.logger import logging
from src.exception import customexception
from src.utils.utils import load_object

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.utils.autologging_utils")

class ModelEvaluation:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def evaluate_model(self, X_test, y_test):
        try:
            logging.info("Evaluating model...")

            # Load the trained model
            model = load_object(self.model_path)
            if model is None:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # Convert X_test to NumPy array to avoid feature name warning
            X_test_array = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

            # Make predictions
            y_pred = model.predict(X_test_array)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Extract metrics for class "Y" safely
            precision_Y = report.get("Y", {}).get("precision", 0.0)
            recall_Y = report.get("Y", {}).get("recall", 0.0)
            f1_score_Y = report.get("Y", {}).get("f1-score", 0.0)

            metrics = {
                "accuracy": accuracy,
                "precision_Y": precision_Y,
                "recall_Y": recall_Y,
                "f1_score_Y": f1_score_Y
            }

            # Log metrics to MLflow
            mlflow.set_experiment("Loan_Prediction_Experiment")
            with mlflow.start_run():
                mlflow.sklearn.autolog()

                if hasattr(model, "get_params"):
                    params = model.get_params()
                    mlflow.log_params(params)

                # Log evaluation metrics manually
                mlflow.log_metrics(metrics)

                logging.info("Model evaluation logged in MLflow.")

            logging.info(f"Evaluation Metrics: {metrics}")
            return metrics

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise customexception(e, sys)


if __name__ == "__main__":
    X_test_path = os.path.join("artifacts", "X_test_transformed.csv")
    y_test_path = os.path.join("artifacts", "y_test.csv")  # Fixed incorrect file name

    try:
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            raise FileNotFoundError("Test data files are missing. Run data transformation first.")

        # Load test data
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()  # Convert to 1D array

        # Evaluate model
        evaluator = ModelEvaluation()
        metrics = evaluator.evaluate_model(X_test, y_test)
        print("Model Evaluation Metrics:", metrics)

    except Exception as e:
        logging.exception("Error in model evaluation script.")
