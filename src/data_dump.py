from src.logger import logging
from src.exception import ProjectException
import pymongo as pm
from src.config import mongo_client  # Import mongo connection from src/config file
import pandas as pd
import numpy as np
import os, sys

# Define database and collection names
database_name = "HR"
collection_name = "Employees"

def dump_data_into_mongodb():
    try:
        # Define path to dataset
        file_path = os.path.join(os.getcwd(), "dataset/HRDataset_v14.csv")
        df = pd.read_csv(file_path)  # Reading dataset as dataframe

        # Convert dataset to dictionary format for MongoDB insertion
        dict_data = df.to_dict(orient="records")
        mongo_client[database_name][collection_name].delete_many({}) # it will delete previous values , to avoid  dupicates values of same dataset.
        mongo_client[database_name][collection_name].insert_many(dict_data)
        logging.info("Dataset successfully inserted into MongoDB database")
    except Exception as e:
        raise ProjectException(e, sys)

if __name__ == "__main__":
    try:
        dump_data_into_mongodb()
        print("Dataset successfully inserted into MongoDB database")
    except Exception as e:
        raise ProjectException(e, sys)
