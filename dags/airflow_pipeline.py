import pandas as pd
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'loan_prediction_pipeline',  
    default_args=default_args,
    description='End-to-end pipeline for loan prediction',
    schedule_interval='@daily',
    catchup=False
)

# Task 1: Data Ingestion
def data_ingestion_task():
    data_ingestion = DataIngestion()
    return data_ingestion.initiate_data_ingestion()

data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion_task,
    dag=dag,
)

# Task 2: Data Transformation
def data_transformation_task():
    data_transformation = DataTransformation()
    train_data_path = "artifacts/train.csv"
    test_data_path = "artifacts/test.csv"

    X_train_transformed, X_test_transformed, y_train, y_test = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

    X_train_transformed_df = pd.DataFrame(X_train_transformed)
    X_test_transformed_df = pd.DataFrame(X_test_transformed)

    X_train_transformed_df.to_csv('artifacts/transformed_X_train.csv', index=False)
    X_test_transformed_df.to_csv('artifacts/transformed_X_test.csv', index=False)
    y_train.to_csv('artifacts/transformed_y_train.csv', index=False)
    y_test.to_csv('artifacts/transformed_y_test.csv', index=False)

    return 'artifacts/transformed_X_train.csv', 'artifacts/transformed_X_test.csv', 'artifacts/transformed_y_train.csv', 'artifacts/transformed_y_test.csv'

data_transformation = PythonOperator(
    task_id='data_transformation',
    python_callable=data_transformation_task,
    dag=dag,
)

# Task 3: Model Training
def model_training_task():
    model_trainer = ModelTrainer()

    # Load the transformed data from CSV files
    X_train_path = "artifacts/transformed_X_train.csv"
    X_test_path = "artifacts/transformed_X_test.csv"
    y_train_path = "artifacts/transformed_y_train.csv"
    y_test_path = "artifacts/transformed_y_test.csv"

    # Read the transformed data into DataFrames
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # Perform model training
    accuracy = model_trainer.initiate_model_training(X_train, X_test, y_train, y_test)
    return accuracy

model_training = PythonOperator(
    task_id='model_training',
    python_callable=model_training_task,
    dag=dag,
)

# Task 4: Model Evaluation
def model_evaluation_task():
    model_evaluator = ModelEvaluation()
    X_test_path = "artifacts/transformed_X_test.csv"
    y_test_path = "artifacts/transformed_y_test.csv"
    
    # Load the transformed data
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()  # Convert to 1D array
    
    # Evaluate the model
    return model_evaluator.evaluate_model(X_test, y_test)

model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation_task,
    dag=dag,
)

# Define task dependencies
data_ingestion >> data_transformation >> model_training >> model_evaluation
