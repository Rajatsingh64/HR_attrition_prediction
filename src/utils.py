import pandas as pd
import numpy as np 
from src.logger import logging
from src.exception import ProjectException
from src.config import mongo_client
import os , sys
import yaml
from src.config import reference_date
import dill



def get_collection_dataframe(database_name:str , collection_name:str)->pd.DataFrame:

    """
    Description : This funtion return collection as Dataframe
    =========================================================
    params:
    database_name : database name:str
    collection_name : collection name:str
    =========================================================
    return Pandas Dataframe of a collection
    """

    try:
        logging.info(f"Reading Dataset from Database {database_name} and Collection {collection_name}")
        df= pd.DataFrame(mongo_client[database_name][collection_name].find())
        logging.info(f"Dataframe column Available {df.columns}")
        if "_id" in df.columns: #removing unwanted column _id from data
            df.drop("_id"  , axis=1 , inplace=True)
        logging.info(f"Rows and Column available {df.shape}")
        return df
    except Exception as e:
        raise ProjectException(e , sys)

def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise ProjectException(e, sys)
    

def convert_columns_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtype == 'object':  # Check if it's a string
                    logging.info(f"Skipping conversion for non-numeric column: {column}")
                    continue  # Skip non-numeric columns

                try:
                    df[column] = df[column].astype('float')
                except ValueError as e:
                    logging.warning(f"Column '{column}' could not be converted to float: {e}")
        return df
    except Exception as e:
        raise e
    
def parse_date_DOB(date_str):
    try:
        dt = pd.to_datetime(date_str, format='%m/%d/%y')
        # If the year is after 2024, assume it's actually 100 years earlier
        if dt.year > 2024:
            dt = dt.replace(year=dt.year - 100)
        return dt
    except:
        try:
            return pd.to_datetime(date_str, format='%m-%d-%Y')
        except:
            print(f"Failed to parse date: {date_str}")  # Debug print
            return pd.NaT

def calculate_age(born):
    if pd.isnull(born):
        print(f"Null birthdate encountered")  # Debug print
        return None
    age = reference_date.year - born.year - ((reference_date.month, reference_date.day) < (born.month, born.day))
    return age

# Function to parse date strings using various formats
def parse_date_for_tenure(date_str):
    if pd.isna(date_str):
        return pd.NaT
    
    # Attempt parsing with multiple formats
    parsed_date = pd.to_datetime(date_str, errors='coerce')
    
    # If year is greater than 2021, adjust it to be 100 years earlier
    if parsed_date.year > 2024:
        parsed_date = parsed_date.replace(year=parsed_date.year - 100)
    
    return parsed_date       

# Function to calculate tenure
def calculate_tenure(row):
    if pd.isna(row['DateofHire']):
        return None
    
    # Calculate tenure based on termination status
    if row['Termd'] == 1 and not pd.isna(row['DateofTermination']):
        tenure = (row['DateofTermination'] - row['DateofHire']).days / 365.25
    else:
        tenure = (reference_date - row['DateofHire']).days / 365.25
    
    return max(tenure, 0)  # Ensure tenure is not negative


def save_object(file_path:str , obj:object)->None:
    try:
        logging.info(f"Entered the Save_Object method of utils")
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)
        logging.info(f"Exited the Save Object of utils")  
    except Exception as e:
        raise ProjectException(e,sys)      
    

def load_object(file_path:str , )-> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f" The file {file_path} not exists")
        with open(file_path , "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise e   
    
def save_numpy_array_data(file_path:str , array:np.array):

    """  
    Save numpy array data to file
    file_path :str location of file to save
    array: np.array data to save
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path , "wb") as file_obj:
            np.save(file_obj , array)
    except Exception as e:
        raise e        
    
def load_numpy_array_data(file_path:str) -> np.array:
    """
    load nump array data from file
    file_path :str location of file to load
    return: np.array data load
    """
    try:
        with open(file_path , "rb")as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ProjectException(e,sys)   