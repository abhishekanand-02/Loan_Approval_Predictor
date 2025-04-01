import pandas as pd
from schema.loan_schema import LoanDataSchema
from pydantic import ValidationError
from src.logger import logging
from src.exception import customexception
import sys

def validate_data(file_path):
    df = pd.read_csv(file_path)
    