from src.logger import logging 
from src.exception import ProjectException 
import os , sys 
from datetime import datetime 


class TrainingPipelineConfig:

    def __init__(self):

        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact" , f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

        except Exception as e:
            raise ProjectException(e , sys)    
        
class DataIngestion:

    def  __init__(self , training_pipeline_config:TrainingPipelineConfig):
        
        try:
            self.database_name ="HR"
            self.collection_name="Employees"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , 'data_ingestion')
            self.feature_store_path = os.path.join(self.data_ingestion_dir , "feature_store , HR_dataset.csv")
            self.train_file_path = os.path.join(self.data_ingestion_dir  , "dataset" , "train.csv")
            self.test_file_path = os.path.join(self.data_ingestion_dir  , "dataset" , "test.csv")
            self.test_size =0.3
        except Exception as e:
            raise ProjectException(e , sys)    


         