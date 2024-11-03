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
        
class DataIngestionConfig:

    def  __init__(self , training_pipeline_config:TrainingPipelineConfig):
        
        try:
            self.database_name ="HR"
            self.collection_name="Employees"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , 'data_ingestion')
            self.feature_store_path = os.path.join(self.data_ingestion_dir , "feature_store", "HR_dataset.csv")
            self.train_file_path = os.path.join(self.data_ingestion_dir  , "dataset" , "train.csv")
            self.test_file_path = os.path.join(self.data_ingestion_dir  , "dataset" , "test.csv")
            self.test_size =0.3
        except Exception as e:
            raise ProjectException(e , sys)    

class DataValidationConfig:

    def __init__(self ,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validation_dir =os.path.join(training_pipeline_config.artifact_dir ,"data_validation")
            self.report_file_path=os.path.join(self.data_validation_dir , "report.yaml")
            self.missing_threshold=0.3 # randomly taking threshold value
            self.base_file_path=os.path.join("dataset/HRDataset_v14.csv") #randomly using main dataset , we need Production Dataset 

        except Exception as e:
            raise ProjectException(e , sys)    

class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        try:
            self.data_transformation_dir=os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
            self.data_transformation_object_path=os.path.join(self.data_transformation_dir , "transformation.pkl")
            self.data_transformation_train_path=os.path.join(self.data_transformation_dir , "transformed" , "train.npz")
            self.data_transformation_test_path=os.path.join(self.data_transformation_dir , "transformed" , "test.npz")
            
        except Exception as e:
            raise ProjectException(e , sys)  

class ModelTrainerConfig:        

    def __init__(self ,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
        self.model_path =os.path.join(self.model_trainer_dir , "model" , "model.pkl")
        self.expected_score = 0.8 
        self.overfiting_threshold=0.1

class ModelEvaluationConfig:

    def __init__(self ,training_pipeline_config:TrainingPipelineConfig ):
             self.change_threshold = 0.1
            

class ModelPusherConfig:
    pass