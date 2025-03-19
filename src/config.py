from dotenv import load_dotenv
from src.logger import logging
from src.exception import ProjectException
from dataclasses import dataclass
import os
import sys
import pandas as pd
import pymongo as pm

#####################
# Load Environment  #
#####################

try:
    print("Loading environment variables from .env file...")
    load_dotenv()
    logging.info("Successfully loaded .env variables.")
except Exception as e:
    raise ProjectException(f"Failed to load .env file: {e}", sys)

##############################
# Environment Configuration  #
##############################

@dataclass
class EnvironmentVariable:
    mongo_url: str = os.getenv("MONGO_DB_URL")

try:
    env = EnvironmentVariable()
    if not env.mongo_url:
        raise ValueError("MONGO_DB_URL is missing in the .env file.")
    logging.info("Environment variables loaded successfully.")
except Exception as e:
    raise ProjectException(e, sys)

##########################
# MongoDB Client Setup   #
##########################

try:
    mongo_client = pm.MongoClient(env.mongo_url)
    logging.info("MongoDB connection established successfully.")
except Exception as e:
    raise ProjectException(f"Failed to connect to MongoDB: {e}", sys)

##########################
# Project Configuration  #
##########################

TARGET_COLUMN = "Termd"

important_features = [
    "Employee_Name", "GenderID", "Salary", "Termd", "Position", "State",
    "DOB", "DateofHire", "DateofTermination", "PerformanceScore",
    "EngagementSurvey", "EmpSatisfaction", "Absences", "ManagerName", "Zip",
    "SpecialProjectsCount", "HispanicLatino", "Department", "MarriedID"
]

reference_date = pd.Timestamp('2024-10-30')

nominal_features = [
    "Employee_Name", "Position", "Department", "ManagerName",
    "PerformanceScore", "State", "HispanicLatino"
]
