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

            model = load_object(self.model_path)
            if model is None:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # Convert X_test to NumPy array to avoid feature name warning
            X_test_array = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

            y_pred = model.predict(X_test_array)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            precision_Y = report.get("Y", {}).get("precision", 0.0)
            recall_Y = report.get("Y", {}).get("recall", 0.0)
            f1_score_Y = report.get("Y", {}).get("f1-score", 0.0)

            metrics = {
                "accuracy": accuracy,
                "precision_Y": precision_Y,
                "recall_Y": recall_Y,
                "f1_score_Y": f1_score_Y
            }

            mlflow.set_experiment("Loan_Prediction_Experiment")
            with mlflow.start_run():
                mlflow.sklearn.autolog()

                if hasattr(model, "get_params"):
                    params = model.get_params()
                    mlflow.log_params(params)

                mlflow.log_metrics(metrics)

                last_model_version = self.get_last_registered_model_version("Loan_Prediction_Model")
                
                # If no model is registered or the new model performs better, register it
                if last_model_version is None or metrics["accuracy"] > last_model_version["accuracy"]:
                    model_name = "Loan_Prediction_Model"
                    result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
                    logging.info(f"Model registered successfully with version {result.version}.")
                else:
                    logging.info("The new model did not perform better, skipping registration.")

                logging.info("Model evaluation logged in MLflow.")

            logging.info(f"Evaluation Metrics: {metrics}")
            return metrics

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise customexception(e, sys)

    def get_last_registered_model_version(self, model_name: str):
        """
        Retrieves the last registered model and its metrics from MLflow.
        Returns a dictionary with the model's performance metrics (if available).
        """
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                logging.info(f"No models registered for {model_name}.")
                return None  

            latest_model_version = model_versions[-1] 
            logging.info(f"Latest registered model version: {latest_model_version.version}")

            model_metrics = latest_model_version.tags 
            accuracy = float(model_metrics.get("accuracy", 0.0)) 

            return {"version": latest_model_version.version, "accuracy": accuracy}

        except Exception as e:
            logging.error(f"Error fetching the last registered model version: {str(e)}")
            raise customexception(e, sys)


if __name__ == "__main__":
    X_test_path = os.path.join("artifacts", "X_test_transformed.csv")
    y_test_path = os.path.join("artifacts", "y_test.csv")  

    try:
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            raise FileNotFoundError("Test data files are missing. Run data transformation first.")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()  
        evaluator = ModelEvaluation()
        metrics = evaluator.evaluate_model(X_test, y_test)
        print("Model Evaluation Metrics:", metrics)

    except Exception as e:
        logging.exception("Error in model evaluation script.")
