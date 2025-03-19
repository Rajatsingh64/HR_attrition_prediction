import os
import sys
import dill
import yaml
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import ProjectException
from src.config import mongo_client, reference_date

##############################
# MongoDB -> DataFrame Utils #
##############################

def get_collection_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Retrieve a collection from MongoDB and convert it to a pandas DataFrame.
    Drops the '_id' column if present.

    Parameters:
        database_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection within the database.

    Returns:
        pd.DataFrame: DataFrame representation of the collection.
    """
    try:
        logging.info(f"Reading dataset from database: {database_name}, collection: {collection_name}")
        df = pd.DataFrame(mongo_client[database_name][collection_name].find())
        logging.info(f"Columns in DataFrame: {df.columns.tolist()}")

        if "_id" in df.columns:
            df.drop("_id", axis=1, inplace=True)
            logging.info("Dropped '_id' column.")

        logging.info(f"DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        raise ProjectException(e, sys)

######################
# YAML File Handling #
######################

def write_yaml_file(file_path: str, data: dict):
    """
    Write a dictionary to a YAML file.

    Parameters:
        file_path (str): Path where the YAML file will be saved.
        data (dict): Dictionary data to write.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
        logging.info(f"YAML file written at: {file_path}")
    except Exception as e:
        raise ProjectException(e, sys)

########################
# DataFrame Converters #
########################

def convert_columns_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Convert all numeric columns (excluding specified ones) to float.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        exclude_columns (list): List of columns to exclude from conversion.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to float.
    """
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtype == 'object':
                    logging.info(f"Skipping conversion for non-numeric column: {column}")
                    continue
                try:
                    df[column] = df[column].astype(float)
                except ValueError as e:
                    logging.warning(f"Column '{column}' could not be converted to float: {e}")
        return df
    except Exception as e:
        raise ProjectException(e, sys)

#####################
# Date Parsing Utils #
#####################

def parse_date_DOB(date_str):
    """
    Parse date of birth string into datetime object, correcting future years.

    Parameters:
        date_str (str): Date string in '%m/%d/%y' or '%m-%d-%Y' format.

    Returns:
        datetime or NaT: Parsed date or NaT if parsing fails.
    """
    try:
        dt = pd.to_datetime(date_str, format='%m/%d/%y')
        if dt.year > 2024:
            dt = dt.replace(year=dt.year - 100)
        return dt
    except:
        try:
            return pd.to_datetime(date_str, format='%m-%d-%Y')
        except:
            logging.warning(f"Failed to parse DOB date: {date_str}")
            return pd.NaT


def calculate_age(born):
    """
    Calculate age from birthdate using reference_date.

    Parameters:
        born (datetime): Birthdate.

    Returns:
        int or None: Calculated age or None if invalid.
    """
    if pd.isnull(born):
        logging.warning("Null birthdate encountered.")
        return None
    age = reference_date.year - born.year - ((reference_date.month, reference_date.day) < (born.month, born.day))
    return age


def parse_date_for_tenure(date_str):
    """
    Parse date string and adjust future years by subtracting 100 years.

    Parameters:
        date_str (str): Date string.

    Returns:
        datetime or NaT: Parsed date or NaT if invalid.
    """
    if pd.isna(date_str):
        return pd.NaT

    parsed_date = pd.to_datetime(date_str, errors='coerce')

    if pd.notna(parsed_date) and parsed_date.year > 2024:
        parsed_date = parsed_date.replace(year=parsed_date.year - 100)

    return parsed_date


def calculate_tenure(row):
    """
    Calculate employee tenure in years.

    Parameters:
        row (pd.Series): DataFrame row with 'DateofHire', 'DateofTermination', and 'Termd'.

    Returns:
        float or None: Tenure in years or None if invalid.
    """
    if pd.isna(row['DateofHire']):
        return None

    if row['Termd'] == 1 and not pd.isna(row['DateofTermination']):
        tenure_days = (row['DateofTermination'] - row['DateofHire']).days
    else:
        tenure_days = (reference_date - row['DateofHire']).days

    tenure_years = tenure_days / 365.25
    return max(tenure_years, 0)

#############################
# Object & Array Serialization #
#############################

def save_object(file_path: str, obj: object) -> None:
    """
    Save an object using dill serialization.

    Parameters:
        file_path (str): File path to save object.
        obj (object): Object to serialize.
    """
    try:
        logging.info(f"Saving object to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object saved successfully.")
    except Exception as e:
        raise ProjectException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a dill-serialized object from file.

    Parameters:
        file_path (str): Path of the file.

    Returns:
        object: Deserialized object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ProjectException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array to file.

    Parameters:
        file_path (str): File path to save array.
        array (np.array): Numpy array to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logging.info(f"Numpy array saved to {file_path}")
    except Exception as e:
        raise ProjectException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array from file.

    Parameters:
        file_path (str): File path of numpy array.

    Returns:
        np.array: Loaded numpy array.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ProjectException(e, sys)
