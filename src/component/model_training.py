from src.entity import artifact_entity,config_entity
from src.exception import ProjectException
from src.logger import logging
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from src.utils import load_numpy_array_data , save_object
import pandas as pd
import numpy as np
import os ,sys


class ModelTrainer:

    def __init__(self , model_trainer_config:config_entity.ModelTrainerConfig , 
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{">"*20} Model Trainer {"<"*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise ProjectException(e,sys)    
        
    def fine_tune_train_best_model(self , X, y):
        try:
            # Set up the grid search for hyperparameter tuning
            param_grid = {
                   'learning_rate': [0.1, 0.2],
                   'max_depth': [3, 5],
                   'n_estimators': [100, 200],
                   'subsample': [0.8, 1.0]}
            # Perform grid search with cross-validation
            xgb_clf= XGBClassifier()
            logging.info(f" HyperPatameter Tuning Started")
            grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X, y)
            # Retrieve the best model from grid search
            best_model = grid_search.best_estimator_
            best_model.fit(X, y)
            logging.info(f"Best model after tuning : {best_model}")
            return best_model
        except Exception as e:
            raise ProjectException(e,sys)   
     
    
    def initiate_model_trainer(self ,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f" Loading Train array and Test array")
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Spliting input and target feature from both train and test array")
            x_train , y_train=train_arr[:,:-1], train_arr[:,-1]
            x_test , y_test = test_arr[:,:-1] , test_arr[:,-1]

            logging.info(f"Train the model")
            model=self.fine_tune_train_best_model(X=x_train , y=y_train)
            logging.info(f"Calculating f1 train score")
            yhat_train=model.predict(x_train)
            f1_train_score=f1_score(y_train,yhat_train)
            
            logging.info(f"Calculating f1 test score")
            yhat_test =model.predict(x_test)
            f1_test_score=f1_score(y_test ,yhat_test)
            logging.info(f"train score: {f1_train_score} test score : {f1_test_score}")
            
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                                 expected accuracy :{self.model_trainer_config.expected_score}: Model actual score : {f1_test_score}")
            
            logging.info(f"checking if our model is overfitting or not")    
            diff =abs(f1_train_score - f1_test_score)

            if diff>self.model_trainer_config.overfiting_threshold:
                raise Exception(f"Train and Test Model Difference : {diff} is more than overfitting Threshold : {self.model_trainer_config.overfiting_threshold}")
            
            #save trained Model
            logging.info(f"Saving model Artifact")
            save_object(file_path=self.model_trainer_config.model_path , obj=model)

            #prepare artifact
            logging.info(f"Prepare artifact")
            model_trainer_artifact =artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path ,
                                                                         f1_train_score=f1_train_score , f1_test_score=f1_test_score )
            logging.info(f"Model Trainer Artifact : {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise ProjectException(e,sys)