from dotenv import load_dotenv 
from src.logger import logging 
from src.exception import ProjectException 
from dataclasses import dataclass
import os , sys
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
