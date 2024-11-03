from src.component.data_ingestion import DataIngestion 
from src.component.data_validation import DataValidation
from src.component.data_transformation import DataTransformation
from src.component.model_training import ModelTrainer
from src.component.model_evaluation import ModelEvaluation
from src.entity.config_entity import TrainingPipelineConfig , DataIngestionConfig ,DataValidationConfig ,DataTransformationConfig , ModelTrainerConfig
from src.entity import config_entity
from src.logger import logging
from src.exception import ProjectException
import os , sys


if __name__=="__main__":

    try:
        #Training Pipeline main Directory artifact
        training_pipeline_config = TrainingPipelineConfig()
        #Data Ingestion (eg. save to artifact folder inside dataset folder
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"{">"*20} Data Ingestion Completed Sucessfully { "<"*20}")
        logging.info(f"{">"*20} Data Ingestion Completed Sucessfully {"<"*20}")
        #Data Validation
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_validation_config , data_ingestion_artifact)
        data_validation_artifact=data_validation.initiate_data_validation()
        print(f"{">"*20} Data Validation Completed Sucessfully {"<"*20}")
        logging.info(f"{">"*20} Data Validation Completed Sucessfully {"<"*20}")


        #data transformation
        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation= DataTransformation(data_transformation_config , data_ingestion_artifact)
        data_tranformation_artifact=data_transformation.initiate_data_transformation()
        print(f"{">"*20} Data Transformation Completed Sucessfully {"<"*20}")
        logging.info(f"{">"*20} Data Transformation Completed Sucessfully {"<"*20}")
        
        #model trainer
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config , data_tranformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        print(f"{">"*20} Model Trainer Triggered Sucessfully {"<"*20}")
        logging.info(f"{">"*20} Model Trainer Triggered Sucessfully {"<"*20}")

        #Model Evaluation 
        model_evaluation_config=config_entity.ModelEvaluationConfig(training_pipeline_config)
        model_evaluation=ModelEvaluation(model_evaluation_config=model_evaluation_config ,data_ingestion_artifact=data_ingestion_artifact ,
                                         data_transformation_artifact=data_tranformation_artifact ,
                                         model_trainer_artifact=model_trainer_artifact)
        model_eva_artifact=model_evaluation.initiate_model_evaluation()
        print(f"{">"*20} Model Evaluation Completed Sucessfully {"<"*20}")
        logging.info(f"{">"*20} Model Evaluation Completed Sucessfully {"<"*20}")

    except Exception as e:
        raise ProjectException(e ,sys)   