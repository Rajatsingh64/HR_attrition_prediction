from src.logger import logging
from src.exception import ProjectException
from src.config import mongo_client  # Importing MongoDB connection setup
import pandas as pd
import os
import sys

# Define database and collection names
DATABASE_NAME = "HR"
COLLECTION_NAME = "Employees"

def dump_data_into_mongodb():
    """
    Description:
    ------------
    Reads the dataset from the local CSV file and dumps it into the specified MongoDB collection.
    """
    try:
        # Define path to dataset
        file_path = os.path.join(os.getcwd(), "dataset", "HRDataset_v14.csv")
        logging.info(f"Reading dataset from {file_path}")
        
        # Reading dataset
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Convert dataset to dictionary format
        dict_data = df.to_dict(orient="records")
        
        # Optional: Remove existing data to avoid duplicates
        # mongo_client[DATABASE_NAME][COLLECTION_NAME].delete_many({})
        
        # Insert data into MongoDB
        mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(dict_data)
        logging.info(f"Dataset successfully inserted into MongoDB: Database - {DATABASE_NAME}, Collection - {COLLECTION_NAME}")
    
    except Exception as e:
        raise ProjectException(e, sys)

#############################
# Main Execution Trigger    #
#############################

if __name__ == "__main__":
    try:
        dump_data_into_mongodb()
        print(f'{"="*20} Dataset successfully inserted into MongoDB! {"="*20}')
    except Exception as e:
        print(f"Error occurred: {e}")
        raise ProjectException(e, sys)
