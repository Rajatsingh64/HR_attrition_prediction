from src.entity import artifact_entity,config_entity
from src.exception import ProjectException
from src.predictor import ModelResolver
from src.logger import logging
from src.entity import artifact_entity
import os ,sys
from src import utils
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from src.config import TARGET_COLUMN , important_features
from imblearn.over_sampling import SMOTE
from src.utils import parse_date_DOB , calculate_age , calculate_tenure ,parse_date_for_tenure

class ModelEvaluation:

    def __init__(self ,
                 model_evaluation_config:config_entity.ModelEvaluationConfig ,
                  data_ingestion_artifact:artifact_entity.DataIngestionArtifact ,
                   data_transformation_artifact:artifact_entity.DataTransformationArtifact ,
                    model_trainer_artifact:artifact_entity.ModelTrainerArtifact ):
        try:
            logging.info(f'{">"*20} Model Evaluation {"<"*20}')
            self.model_evaluation_config=model_evaluation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact= model_trainer_artifact
            self.model_resovler=ModelResolver()
        except Exception as e:
            raise ProjectException(e ,sys)    
        
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
         #if Saved model folder has model , we compare production model with new one
         #which model is best trained or model from the saved model folder
        
        try:
            logging.info(f"if saved model folder has model then we will compare , which model is well trained")
            latest_dir_path=self.model_resovler.get_latest_dir_path()
            if latest_dir_path==None:
                model_evaluation_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True ,improved_accuracy=None)
                logging.info(f"Model Evaluation artifact : {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            logging.info(f"Finding Location of Transformer model")
            transformer_path = self.model_resovler.get_latest_transformer_path()
            model_path = self.model_resovler.get_latest_model_path()

            logging.info(f"Previous Trained objects of Transformer and Model")
            transformer= utils.load_object(file_path=transformer_path)
            model= utils.load_object(file_path=model_path)

            logging.info(f"Currently Trained Model Objects")
            #currently trrained model objects
            current_model=utils.load_object(file_path=self.model_trainer_artifact.model_path)
            current_transformer=utils.load_object(file_path=self.data_transformation_artifact.transformation_object_path)
            test_df =pd.read_csv(self.data_ingestion_artifact.test_file_path)
            not_important_features = [ column for column in test_df.columns if column not in important_features]
            test_df.drop(not_important_features, axis=1 , inplace=True) #Lets drop unnecessary column
            
            test_df["DOB"] = test_df["DOB"].apply(parse_date_DOB)
            test_df['Age'] = test_df['DOB'].apply(calculate_age)
            # Replace ages greater than 50 with the mean age
            mean_age = test_df['Age'][test_df['Age'] <= 50].mean()
            test_df['Age'] = test_df['Age'].apply(lambda x: mean_age if x > 50 else x)
            logging.info(f"Calculating Age using DOB column for Test df")

            test_df['DateofHire'] = test_df['DateofHire'].apply(parse_date_for_tenure)
            test_df['DateofTermination'] = test_df['DateofTermination'].apply(parse_date_for_tenure)
            test_df['Tenure'] = test_df.apply(calculate_tenure, axis=1)
            test_df.drop(["DOB" , "DateofHire" ,"DateofTermination"] ,axis=1 ,inplace=True)
            logging.info(f"Calculated employees Tenure for test df")
           
            # Input and target data 
            input_test_df=test_df.drop("Termd" , axis=1)
            target_df=test_df[TARGET_COLUMN] #Already encoded
            
            #Follow Same Process used in a data transformation and Training , to aviod error such as (eg.shape error)
            #Features of previous transfomer model , but i dont have old or Production model.i am using same model new one for both
            input_features_name=list(transformer.feature_names_in_)  #features name transformed using encoder
            test_encoded=transformer.transform(input_test_df[input_features_name]) #tranform array value
            test_df_encoded=pd.DataFrame(test_encoded , columns=transformer.get_feature_names_out(input_features_name)) #creating dataframe of encoded features
            input_df=pd.concat([input_test_df.drop(columns=input_features_name).reset_index(drop=True), test_df_encoded.reset_index(drop=True)], axis=1) #combining both main features with encoded one and remove previous column encoded one
            #Prediction using old model
            y_pred=model.predict(input_df)
            previous_model_score=f1_score(target_df , y_pred)
            logging.info(f"Accuracy using Previous Model Training : {previous_model_score}")

            #Accuracy using Current Model Training
            input_feature_name=list(current_transformer.feature_names_in_) #Using New transformer pkl  
            input_encoded_features=current_transformer.transform(input_test_df[input_feature_name])
            test_df_encoded=pd.DataFrame(input_encoded_features , columns=transformer.get_feature_names_out(input_features_name))
            input_df=pd.concat([input_test_df.drop(columns=input_features_name).reset_index(drop=True), test_df_encoded.reset_index(drop=True)], axis=1)
            #Prediction using current or new Model
            y_pred=current_model.predict(input_df)
            current_model_score=f1_score(target_df, y_pred)
            logging.info(f"Accuracy using Current Model Training : {current_model_score}")
            if current_model_score<previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")
        
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise ProjectException(e ,sys) 
