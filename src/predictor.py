import os , sys
from glob import glob 
from typing import Optional
from src.logger import logging
from src.exception import ProjectException

file_name = "HR_dataset.csv"
train_file_name="train.csv"
test_file_name="test.csv"
transformer_object_file_name="transformer.pkl"
model_file_name="model.pkl"

class ModelResolver:

    def __init__(self , model_registry:str ="saved_models" , 
                 transformer_dir_name="transformer" , 
                 model_dir_name="model"):
        try:
            self.model_registry=model_registry
            os.makedirs(self.model_registry , exist_ok=True)
            self.transformer_dir_name=transformer_dir_name
            self.model_dir_name=model_dir_name
        except Exception as e:
            raise ProjectException(e,sys)  
        
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_name=os.listdir(self.model_registry)
            if len(dir_name)==0:
                return None
            dir_name=list(map(int,dir_name))
            lates_dir_name=max(dir_name)
            return os.path.join(self.model_registry ,f"{lates_dir_name}")
        except Exception as e:
            raise ProjectException(e,sys)
        
    def get_latest_model_path(self)   :
        try:
            latest_dir= self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not Available")
            return os.path.join(latest_dir,self.model_dir_name , model_file_name)
        except Exception as e:
            raise e
        
    def get_latest_transformer_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not Available")
            return os.path.join(latest_dir,self.transformer_dir_name,transformer_object_file_name)
        except Exception as e:
            raise e
        
    
    def get_latest_save_dir_path(self)->str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir==None:
                return os.path.join(self.model_registry,f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry,f"{latest_dir_num+1}")
        except Exception as e:
            raise e

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,model_file_name)
        except Exception as e:
            raise e

    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name,transformer_object_file_name)
        except Exception as e:
            raise e

    

class Predictor:

    def __init__(self,model_resolver:ModelResolver):
        self.model_resolver=model_resolver
      
