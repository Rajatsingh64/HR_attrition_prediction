from src.predictor import ModelResolver
from src.entity.config_entity import ModelPusherConfig
from src.exception import ProjectException
from src.utils import load_object, save_object
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact
import os, sys


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher Initialization {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_pusher(self,) -> ModelPusherArtifact:
        try:
            # Create necessary directories
            logging.info(f"Creating directories if not exist")
            os.makedirs(self.model_pusher_config.pusher_model_dir, exist_ok=True)
            os.makedirs(self.model_pusher_config.saved_model_dir, exist_ok=True)

            # Load transformer, model, and target encoder
            logging.info(f"Loading transformer, model  objects")
            transformer = load_object(file_path=self.data_transformation_artifact.transformation_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
          
           # Save to model pusher directory
            logging.info(f"Saving model components to model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            logging.info(f"Transformer saved at {self.model_pusher_config.pusher_transformer_path}")

            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            logging.info(f"Model saved at {self.model_pusher_config.pusher_model_path}")

           # Save to saved model directory with versioning
            logging.info(f"Saving model components to versioned saved model directory")

            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            
            save_object(file_path=transformer_path, obj=transformer)
            logging.info(f"Transformer version saved at {transformer_path}")

            save_object(file_path=model_path, obj=model)
            logging.info(f"Model version saved at {model_path}")

            # Prepare Model Pusher Artifact
            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            logging.info(f"Model Pusher Artifact created: {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise ProjectException(e, sys)
