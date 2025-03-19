import os, sys
from typing import Optional
from src.logger import logging
from src.exception import ProjectException

# File names
file_name = "HR_dataset.csv"
train_file_name = "train.csv"
test_file_name = "test.csv"
transformer_object_file_name = "transformer.pkl"
model_file_name = "model.pkl"
target_encoder_file_name = "target_encoder.pkl"  # Newly added for target encoder

class ModelResolver:
    def __init__(self, model_registry: str = "saved_models",
                 transformer_dir_name: str = "transformer",
                 model_dir_name: str = "model",
                 target_encoder_dir_name: str = "target_encoder"):  # New encoder dir
        try:
            self.model_registry = model_registry
            self.transformer_dir_name = transformer_dir_name
            self.model_dir_name = model_dir_name
            self.target_encoder_dir_name = target_encoder_dir_name
            os.makedirs(self.model_registry, exist_ok=True)
            logging.info(f"Initialized ModelResolver with registry: {self.model_registry}")
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_dir_path(self) -> Optional[str]:
        """Returns latest saved model directory path"""
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir = max(dir_names)
            return os.path.join(self.model_registry, f"{latest_dir}")
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_model_path(self) -> str:
        """Returns latest model path"""
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise FileNotFoundError("No saved model available.")
            return os.path.join(latest_dir, self.model_dir_name, model_file_name)
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_transformer_path(self) -> str:
        """Returns latest transformer path"""
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise FileNotFoundError("No saved transformer available.")
            return os.path.join(latest_dir, self.transformer_dir_name, transformer_object_file_name)
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_target_encoder_path(self) -> str:
        """Returns latest target encoder path"""
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise FileNotFoundError("No saved target encoder available.")
            return os.path.join(latest_dir, self.target_encoder_dir_name, target_encoder_file_name)
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_save_dir_path(self) -> str:
        """Returns path to save next model version"""
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                new_dir = 0
            else:
                latest_dir_num = int(os.path.basename(latest_dir))
                new_dir = latest_dir_num + 1
            return os.path.join(self.model_registry, f"{new_dir}")
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_save_model_path(self) -> str:
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, model_file_name)
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_save_transformer_path(self) -> str:
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name, transformer_object_file_name)
        except Exception as e:
            raise ProjectException(e, sys)

    def get_latest_save_target_encoder_path(self) -> str:
        """Returns path to save target encoder"""
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.target_encoder_dir_name, target_encoder_file_name)
        except Exception as e:
            raise ProjectException(e, sys)
