from dotenv import load_dotenv 
from src.logger import logging 
from src.exception import ProjectException 
from dataclasses import dataclass
import os , sys
import pandas as pd
import pymongo as pm


# To hide sensitive infomation such as pass and url , using dot env file 
print("Loading  .env ")
load_dotenv()
logging.info(f"loading .env ")


@dataclass
class EnvironmentVariable:
    mongo_url:str = os.getenv("mongo_url")  #Loading mongo url from dot env file

env = EnvironmentVariable() #instance

mongo_client = pm.MongoClient(env.mongo_url) # Establish connection between MongoDb database
logging.info(f"Connected to MongoDb ")

#selecting impotant feature
TARGET_COLUMN ="Termd"
#selecting impotant feature for model training
important_features=["Employee_Name" , 
                    "GenderID" ,"Salary" ,
                    "Termd" ,"Position" ,
                    "State" ,"DOB" ,"DateofHire" ,
                    "DateofTermination" ,
                    "PerformanceScore" ,
                    "EngagementSurvey" ,
                    "EmpSatisfaction" ,
                    "Absences" , "ManagerName", "Zip" ,
                    "SpecialProjectsCount" ,
                    "HispanicLatino","Department" ,
                    'MarriedID']
# Define a reference date for tenure calculation
reference_date = pd.Timestamp('2024-10-30')
