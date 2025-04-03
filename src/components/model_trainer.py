import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import customexception
from src.utils.utils import save_object

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_X, test_X, train_y, test_y):
        try:
            logging.info("Initializing model training...")

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            lr_model = LogisticRegression(max_iter=1000, random_state=42)

            logging.info("Training the Random Forest model...")
            rf_model.fit(train_X, train_y.ravel())

            logging.info("Training the Logistic Regression model...")
            lr_model.fit(train_X, train_y.ravel())

            logging.info("Making predictions with the Random Forest model...")
            rf_predictions = rf_model.predict(test_X)

            logging.info("Making predictions with the Logistic Regression model...")
            lr_predictions = lr_model.predict(test_X)

            rf_accuracy = accuracy_score(test_y, rf_predictions)
            lr_accuracy = accuracy_score(test_y, lr_predictions)

            logging.info(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
            logging.info(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

            if rf_accuracy > lr_accuracy:
                best_model = rf_model
                best_accuracy = rf_accuracy
                logging.info("Random Forest model performed better. Saving Random Forest model.")
            elif rf_accuracy < lr_accuracy:
                best_model = lr_model
                best_accuracy = lr_accuracy
                logging.info("Logistic Regression model performed better. Saving Logistic Regression model.")
            else:
                best_model = rf_model
                best_accuracy = rf_accuracy
                logging.info("Both models have the same accuracy. Preferring Random Forest model and saving it.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")

            return best_accuracy

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    logging.info("Starting the data ingestion process...")
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    logging.info("Starting the data transformation process...")
    data_transformation = DataTransformation()
    train_X, test_X, train_y, test_y = data_transformation.initialize_data_transformation(train_path, test_path)

    logging.info("Starting the model training process...")
    model_trainer = ModelTrainer()
    accuracy = model_trainer.initiate_model_training(train_X, test_X, train_y, test_y)

    logging.info(f"Training pipeline completed successfully with accuracy: {accuracy * 100:.2f}%")
