import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import customexception
from src.utils.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    X_train_path = os.path.join("artifacts", "X_train_transformed.csv")
    X_test_path = os.path.join("artifacts", "X_test_transformed.csv")
    y_train_path = os.path.join("artifacts", "y_train.csv")
    y_test_path = os.path.join("artifacts", "y_test.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        """Creates preprocessing pipeline for numerical and categorical features."""
        try:
            logging.info('Initializing Data Transformation')

            # Define numerical and categorical columns
            numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

            logging.info('Setting up transformation pipelines')

            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', numerical_pipeline, numerical_columns),
                ('cat_pipeline', categorical_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformation: {str(e)}")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        """Loads train and test data, applies preprocessing, and saves transformed data."""
        try:
            logging.info("Loading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Successfully loaded train and test data.")

            preprocessing_pipeline = self.get_data_transformation()

            target_column = 'Loan_Status'
            columns_to_drop = ['Loan_ID', target_column]

            # Separate features (X) and target (y)
            X_train = train_df.drop(columns=columns_to_drop, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=columns_to_drop, axis=1)
            y_test = test_df[target_column]

            # Apply transformations
            X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
            X_test_transformed = preprocessing_pipeline.transform(X_test)

            logging.info("Data transformation completed successfully.")

            # Save transformed datasets
            pd.DataFrame(X_train_transformed).to_csv(self.data_transformation_config.X_train_path, index=False)
            pd.DataFrame(X_test_transformed).to_csv(self.data_transformation_config.X_test_path, index=False)
            y_train.to_csv(self.data_transformation_config.y_train_path, index=False)
            y_test.to_csv(self.data_transformation_config.y_test_path, index=False)
            logging.info("Transformed data saved successfully.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_pipeline
            )
            logging.info("Preprocessing object saved successfully.")

            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            logging.error(f"Exception in initialize_data_transformation: {str(e)}")
            raise customexception(e, sys)


# if __name__ == "__main__":
#     train_data_path = 'artifacts/train.csv'
#     test_data_path = 'artifacts/test.csv'

#     data_transformation_instance = DataTransformation()
#     train_X, test_X, train_y, test_y = data_transformation_instance.initialize_data_transformation(train_data_path, test_data_path)
