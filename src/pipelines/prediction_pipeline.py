import os
import sys
import pandas as pd
from src.exception import customexception 
from src.logger import logging
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        logging.info("Initializing the prediction pipeline object...")

    def predict(self, features: pd.DataFrame, target: pd.Series = None):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading preprocessor and model from the artifacts directory...")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Applying preprocessing to the input data...")
            scaled_features = preprocessor.transform(features)
            predictions = model.predict(scaled_features)

            logging.info("Prediction process completed successfully.")
            return predictions

        except Exception as e:
            logging.error("An error occurred during prediction.")
            raise customexception(e, sys)


class CustomData:
    def __init__(self, applicant_income: float, coapplicant_income: float, loan_amount: float, 
                 loan_amount_term: float, credit_history: float, gender: str, married: str, 
                 dependents: str, education: str, self_employed: str, property_area: str):
        
        self.applicant_income = applicant_income
        self.coapplicant_income = coapplicant_income
        self.loan_amount = loan_amount
        self.loan_amount_term = loan_amount_term
        self.credit_history = credit_history
        self.gender = gender
        self.married = married
        self.dependents = dependents
        self.education = education
        self.self_employed = self_employed
        self.property_area = property_area

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data_dict = {
                'ApplicantIncome': [self.applicant_income],
                'CoapplicantIncome': [self.coapplicant_income],
                'LoanAmount': [self.loan_amount],
                'Loan_Amount_Term': [self.loan_amount_term],
                'Credit_History': [self.credit_history],
                'Gender': [self.gender],
                'Married': [self.married],
                'Dependents': [self.dependents],
                'Education': [self.education],
                'Self_Employed': [self.self_employed],
                'Property_Area': [self.property_area]
            }

            df = pd.DataFrame(data_dict)
            logging.info("Dataframe created successfully.")
            return df
        except Exception as e:
            logging.error("Exception occurred while creating the dataframe.")
            raise customexception(e, sys)


# if __name__ == "__main__":
#     custom_data = CustomData(
#         gender="Male",
#         married="Yes",
#         dependents="1",
#         education="Graduate",
#         self_employed="No",
#         applicant_income=4583,
#         coapplicant_income=1508,
#         loan_amount=128,
#         loan_amount_term=360,
#         credit_history=1,
#         property_area="Rural"
#     )

#     data = custom_data.get_data_as_dataframe()

#     prediction_pipeline = PredictPipeline()
#     predictions = prediction_pipeline.predict(features=data)
    
#     for pred in predictions:
#         print(f"Prediction: {pred}")
