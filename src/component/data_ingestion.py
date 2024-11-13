from src.logger import logging 
from src.exception  import ProjectException 
from src.config import mongo_client 
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils import get_collection_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import os , sys


class DataIngestion:

    def __init__(self, data_ingestion_config:DataIngestionConfig):
        
        try:
            logging.info(f'{">"*20} Data Ingestion {"<"*20}')
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise ProjectException(e,sys)    


    def initiate_data_ingestion(self)->DataIngestionArtifact:

        try:
            logging.info(f"Exporting Data as Dataframe")
            #Exporting dataset as Dataframe from MongoDB Atlas
            df:pd.DataFrame = get_collection_dataframe(
                             database_name=self.data_ingestion_config.database_name,
                             collection_name=self.data_ingestion_config.collection_name
            )
            #Lets save this dataframe into feature store folder
            logging.info(f"Create feature store folder if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_path)
            os.makedirs(feature_store_dir , exist_ok=True)
            logging.info(f"Saving Dataset to feature store folder")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_path , index=False , header=True)
            
            logging.info(f"Lets split our data into train test dataframe")
            #spliting dataset into train test
            train_df , test_df = train_test_split(df , test_size=self.data_ingestion_config.test_size ,random_state=10)

            #lets create directory dataset to save train df and test df if not avialable
            logging.info(f"create directory dataset not avialable")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir , exist_ok=True)
            
            logging.info(f"Save train and test df to feature store folder")
            train_df.to_csv(self.data_ingestion_config.train_file_path , index=False , header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False , header=True)
            
            #Lets prepare artifact or outputs save inside artifact folder
            data_ingestion_artifact=DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_path ,
                train_file_path= self.data_ingestion_config.train_file_path ,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data Ingestion Artifact : {data_ingestion_artifact}")
            return data_ingestion_artifact            
        except Exception as e:
            raise ProjectException (e , sys)
