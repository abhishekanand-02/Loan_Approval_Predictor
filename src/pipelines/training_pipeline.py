import os
import sys
import luigi,shutil
import pandas as pd
import json
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer  
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
from src.exception import customexception

class DataIngestionTask(luigi.Task):
    """
    task for Data Ingestion.
    It saves train and test datasets as outputs.
    """
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")

    def output(self):
        return {
            "train": luigi.LocalTarget(self.train_data_path),
            "test": luigi.LocalTarget(self.test_data_path),
        }

    def run(self):
        """Execute data ingestion."""
        try:
            logging.info("Step 1: Ingesting data triggered from the training pipeline...")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Write the datasets to the output targets
            train_data.to_csv(self.output()["train"].path, index=False)
            test_data.to_csv(self.output()["test"].path, index=False)

            logging.info(f"Data ingestion completed successfully. Train: {train_path}, Test: {test_path}")

        except Exception as e:
            logging.exception("Exception occurred during data ingestion task.")
            raise customexception(e)

class DataTransformationTask(luigi.Task):
    """
    task for Data Transformation.
    It transforms train and test datasets and saves the preprocessing object.
    """
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def requires(self):
        """Define dependencies."""
        return DataIngestionTask() 

    def output(self):
        """Define output targets."""
        return {
            "preprocessor": luigi.LocalTarget(self.preprocessor_path),
            "X_train": luigi.LocalTarget(os.path.join("artifacts", "X_train_transformed.csv")),
            "X_test": luigi.LocalTarget(os.path.join("artifacts", "X_test_transformed.csv")),
            "y_train": luigi.LocalTarget(os.path.join("artifacts", "y_train.csv")),
            "y_test": luigi.LocalTarget(os.path.join("artifacts", "y_test.csv"))
        }

    def run(self):
        """Execute data transformation."""
        try:
            logging.info("Step 2: Transforming data...")
            data_transformation = DataTransformation()
            
            # Initialize data transformation and get transformed data
            train_X, test_X, train_y, test_y = data_transformation.initialize_data_transformation(
                self.train_data_path,
                self.test_data_path
            )
            
            pd.DataFrame(train_X).to_csv(self.output()["X_train"].path, index=False)
            pd.DataFrame(test_X).to_csv(self.output()["X_test"].path, index=False)
            pd.Series(train_y).to_csv(self.output()["y_train"].path, index=False)
            pd.Series(test_y).to_csv(self.output()["y_test"].path, index=False)

            logging.info("Data transformation completed successfully.")

        except Exception as e:
            logging.exception("Exception occurred during data transformation task.")
            raise customexception(e)


class ModelTrainerTask(luigi.Task):
    """
    Luigi task for training the model.
    """
    model_path = os.path.join("artifacts", "model.pkl")

    def requires(self):
        """Define dependencies."""
        return DataTransformationTask()

    def output(self):
        """Define output target."""
        return luigi.LocalTarget(self.model_path)

    def run(self):
        """Execute model training."""
        try:
            logging.info("Step 3: Training the model...")

            # Load transformed datasets from dependencies
            transformation_output = self.requires().output()
            train_X = pd.read_csv(transformation_output["X_train"].path)
            test_X = pd.read_csv(transformation_output["X_test"].path)
            train_y = pd.read_csv(transformation_output["y_train"].path).squeeze()
            test_y = pd.read_csv(transformation_output["y_test"].path).squeeze()

            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_X, test_X, train_y, test_y)

            logging.info("Model training completed successfully.")

        except Exception as e:
            logging.exception("Exception occurred during model training task.")
            raise customexception(e, sys) 



class ModelEvaluationTask(luigi.Task):
    """
    Task for evaluating the trained model.
    Saves the evaluation metrics as output.
    """
    
    evaluation_report_path = os.path.join("artifacts", "evaluation.json")
    X_test_path = os.path.join("artifacts", "X_test_transformed.csv")
    y_test_path = os.path.join("artifacts", "y_test.csv")

    def requires(self):
        """Define dependencies."""
        return ModelTrainerTask()

    def output(self):
        """Define output target."""
        return luigi.LocalTarget(self.evaluation_report_path)

    def load_test_data(self):
        """Load transformed test data from CSV files."""
        if not os.path.isfile(self.X_test_path):
            raise FileNotFoundError(f"Test data file missing: {self.X_test_path}")
        if not os.path.isfile(self.y_test_path):
            raise FileNotFoundError(f"Test data file missing: {self.y_test_path}")

        try:
            X_test = pd.read_csv(self.X_test_path)
            y_test = pd.read_csv(self.y_test_path).values.ravel() 
            return X_test, y_test
        except Exception as e:
            logging.exception("Error loading test data.")
            raise e

    def run(self):
        """Execute model evaluation."""
        try:
            logging.info("Step 4: Evaluating the model...")

            X_test, y_test = self.load_test_data()

            model_evaluator = ModelEvaluation()
            evaluation_metrics = model_evaluator.evaluate_model(X_test, y_test)

            with open(self.output().path, "w", encoding="utf-8") as f:
                json.dump(evaluation_metrics, f, indent=4)

            logging.info("Model evaluation completed successfully.")

        except Exception as e:
            logging.exception("Exception occurred during model evaluation task.")
            raise customexception(e, sys)


if __name__ == "__main__":
    luigi.build([ModelEvaluationTask()],workers = 1, local_scheduler=False)



# Below code is working fine , it is the simple python script which act as a pipeline for all the components.

    # import os
    # import sys
    # from src.components.data_ingestion import DataIngestion
    # from src.components.data_transformation import DataTransformation
    # from src.components.model_trainer import ModelTrainer
    # from src.components.model_evaluation import ModelEvaluation
    # from src.logger import logging
    # from src.exception import customexception

    # class TrainPipeline:
    #     def __init__(self):
    #         self.data_ingestion = DataIngestion()
    #         self.data_transformation = DataTransformation()
    #         self.model_trainer = ModelTrainer()
    #         self.model_evaluation = ModelEvaluation()  

    #     def run_pipeline(self):
    #         try:
    #             logging.info("Starting the training pipeline...")

    #             # Step 1: Data Ingestion
    #             logging.info("Step 1: Ingesting data...")
    #             train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

    #             # Step 2: Data Transformation
    #             logging.info("Step 2: Transforming data...")
    #             train_X, test_X, train_y, test_y = self.data_transformation.initialize_data_transformation(
    #                 train_data_path, test_data_path
    #             )

    #             # Step 3: Model Training
    #             logging.info("Step 3: Training the model...")
    #             accuracy = self.model_trainer.initiate_model_training(train_X, test_X, train_y, test_y)

    #             # Step 4: Model Evaluation 
    #             logging.info("Step 4: Evaluating the model...")
    #             evaluation_metrics = self.model_evaluation.evaluate_model(test_X, test_y)

    #             logging.info(f"Training pipeline completed successfully with accuracy: {accuracy * 100:.2f}%")

    #         except Exception as e:
    #             logging.error(f"Error in training pipeline: {str(e)}")
    #             raise customexception(e, sys)

    # if __name__ == "__main__":
    #     pipeline = TrainPipeline()
    #     pipeline.run_pipeline()



# class ModelTrainerTask(luigi.Task):
#     """
#     Task for Model Training.
#     It trains a machine learning model and saves it.
#     """
#     trained_model_path = os.path.join("artifacts", "model.pkl")

#     def requires(self):
#         """Define dependencies."""
#         return DataTransformationTask()

#     def output(self):
#         """Define output targets."""
#         return luigi.LocalTarget(self.trained_model_path)

#     def run(self):
#         """Execute model training."""
#         try:
#             logging.info("Step 3: Model training started...")

#             # Load transformed data
#             train_X = pd.read_csv(self.input()["X_train"].path).values
#             test_X = pd.read_csv(self.input()["X_test"].path).values
#             train_y = pd.read_csv(self.input()["y_train"].path).values
#             test_y = pd.read_csv(self.input()["y_test"].path).values

#             model_trainer = ModelTrainer()
#             accuracy = model_trainer.initiate_model_training(train_X, test_X, train_y, test_y)

#             logging.info(f"Model training completed successfully with accuracy: {accuracy:.2f}")

#             # Save the trained model
#             with open(self.output().path, "wb") as model_file:
#                 model_file.write(open(self.trained_model_path, "rb").read())

#         except Exception as e:
#             logging.exception("Exception occurred during model training task.")
#             raise customexception(e)

