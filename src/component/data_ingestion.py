from src.logger import logging
from src.exception import ProjectException
from src.config import mongo_client
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils import get_collection_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd
import os, sys


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f'{"="*20} Data Ingestion Started {"="*20}')
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # Step 1: Export data from MongoDB
            logging.info(f"Exporting dataset from MongoDB collection as DataFrame")
            df: pd.DataFrame = get_collection_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name,
            )
            logging.info(f"Dataframe Shape: {df.shape}")

            # Step 2: Save raw data to feature store
            logging.info(f"Creating feature store directory if not exists")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info(f"Saving dataset to feature store at: {self.data_ingestion_config.feature_store_path}")
            df.to_csv(self.data_ingestion_config.feature_store_path, index=False, header=True)

            # Step 3: Split into train-test
            logging.info(f"Splitting data into Train and Test sets")
            train_df, test_df = train_test_split(
                df,
                test_size=self.data_ingestion_config.test_size,
                random_state=10,
            )
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Step 4: Save train & test datasets
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info(f"Saving Train dataset at: {self.data_ingestion_config.train_file_path}")
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)

            logging.info(f"Saving Test dataset at: {self.data_ingestion_config.test_file_path}")
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            # Step 5: Prepare artifact
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )

            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            logging.info(f'{"="*20} Data Ingestion Completed {"="*20}')
            return data_ingestion_artifact

        except Exception as e:
            raise ProjectException(e, sys)
