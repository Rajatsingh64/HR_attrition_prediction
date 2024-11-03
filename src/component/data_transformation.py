from src.entity import artifact_entity,config_entity
from src.exception import ProjectException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os ,sys
from src.config import TARGET_COLUMN , important_features
from src.utils import calculate_age , calculate_tenure , parse_date_DOB ,parse_date_for_tenure , save_numpy_array_data , save_object
 

class DataTransformation:

    def __init__(self , 
                 data_transformation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{">"*20} Data Transformation {"<"*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise ProjectException(e , sys)   

    
    
    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:

        try:
            logging.info(f"Reading Train and Test data as Dataframe")
            #Reading dataset as Dataframe
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Lets drop Unnecessary Features
            not_important_features = [ column for column in train_df.columns if column not in important_features]
            train_df.drop(not_important_features, axis=1 , inplace=True) #Lets drop unnecessary column 
            test_df.drop(not_important_features, axis=1 , inplace=True) #Lets drop unnecessary column 
            
            logging.info(f"Droping Unnecessary columns")
            #Lets calcualte age and tenure
            logging.info(f"Coverting DOB to Pd.datetime format for both Train and Test df")
            train_df["DOB"] = train_df["DOB"].apply(parse_date_DOB)
            test_df["DOB"] = test_df["DOB"].apply(parse_date_DOB)
            
            #lets Calculate Age for train_df
            train_df['Age'] = train_df['DOB'].apply(calculate_age)
            logging.info(f"Calculating Age using DOB column for Both Train and Test df")
            # Replace ages greater than 50 with the mean age
            mean_age = train_df['Age'][train_df['Age'] <= 50].mean()
            train_df['Age'] = train_df['Age'].apply(lambda x: mean_age if x > 50 else x)
            #Lets Calculate Age for train_df
            test_df['Age'] = test_df['DOB'].apply(calculate_age)
            test_df['Age'] = test_df['Age'].apply(lambda x: mean_age if x > 50 else x)
            logging.info(f"Replacing Age greater than 50 year with Average Age of an column")
            
            logging.info(f"Started calculating Tenure using Date of hire and Termination")
            #Lets Calculate Tenure
            # Apply the parsing function to the DateofHire and DateofTermination columns
            train_df['DateofHire'] = train_df['DateofHire'].apply(parse_date_for_tenure)
            train_df['DateofTermination'] = train_df['DateofTermination'].apply(parse_date_for_tenure)
            # Apply the tenure calculation function
            train_df['Tenure'] = train_df.apply(calculate_tenure, axis=1)
            
            test_df['DateofHire'] = test_df['DateofHire'].apply(parse_date_for_tenure)
            test_df['DateofTermination'] = test_df['DateofTermination'].apply(parse_date_for_tenure)
            # Apply the tenure calculation function
            test_df['Tenure'] = test_df.apply(calculate_tenure, axis=1)
            #Lets drop Unnecessary column 
            train_df.drop(["DOB" , "DateofHire" ,"DateofTermination"],axis=1 ,inplace=True)
            test_df.drop(["DOB" , "DateofHire" ,"DateofTermination"] ,axis=1 ,inplace=True)
            logging.info(f"Calculated employees Tenure for both Train and test df")
            
           
            nominal_features = ['Employee_Name', 'Position', 'Department', 'ManagerName', 'PerformanceScore' ,"State" ,"HispanicLatino"]
            logging.info(f"Data Transformation started for: {nominal_features}")
            one_hot_encoder=OneHotEncoder(sparse_output=False , handle_unknown="ignore")
            
            #Fit and transform on train data
            train_encoded_features = one_hot_encoder.fit_transform(train_df[nominal_features])
            train_encoded_df = pd.DataFrame(train_encoded_features, columns=one_hot_encoder.get_feature_names_out(nominal_features))
            

            # Transform on test data
            test_encoded_features = one_hot_encoder.transform(test_df[nominal_features])
            test_encoded_df = pd.DataFrame(test_encoded_features, columns=one_hot_encoder.get_feature_names_out(nominal_features))
            
            logging.info(f"Data tranformed for this features :{nominal_features}")

            # Concatenate the original DataFrame (without the nominal features) with the new encoded DataFrame
            train_df = pd.concat([train_df.drop(columns=nominal_features).reset_index(drop=True), train_encoded_df.reset_index(drop=True)], axis=1)
            test_df = pd.concat([test_df.drop(columns=nominal_features).reset_index(drop=True), test_encoded_df.reset_index(drop=True)], axis=1)

            #Select Input Features for both train and test data
            input_features_train_df=train_df.drop(TARGET_COLUMN , axis=1)
            input_features_test_df=test_df.drop(TARGET_COLUMN , axis=1)
            logging.info(f"Spliting inpur features for both Train and Test data")
            #Selecting target features for both train and test data
            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_test_df=test_df[TARGET_COLUMN]
            logging.info(f"spliting target features for train and test data")
            #Lets Balance imbalance data
            smote=SMOTE()
            logging.info(f"Before resampling in training set input :{input_features_train_df.shape} Target : {target_feature_train_df.shape}")
            #apply to Training data
            input_features_train_df , target_feature_train_df =smote.fit_resample(input_features_train_df,target_feature_train_df)
            logging.info(f"After resampling in training set input :{input_features_train_df.shape} Target : {target_feature_train_df.shape}")
            #apply to test data
            logging.info(f"Before resampling in testing set input :{input_features_test_df.shape} Target : {target_feature_test_df.shape}")
            input_features_test_df , target_feature_test_df =smote.fit_resample(input_features_test_df,target_feature_test_df)
            logging.info(f"After resampling in testing set input :{input_features_test_df.shape} Target : {target_feature_test_df.shape}")

            #target encoder
            train_arr = np.c_[input_features_train_df,target_feature_train_df]
            test_arr=np.c_[input_features_test_df,target_feature_test_df]
            
            #save numpy array 
            save_numpy_array_data(file_path=self.data_transformation_config.data_transformation_train_path , array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.data_transformation_test_path , array=test_arr)
            
            #input feature  encoder object
            save_object(self.data_transformation_config.data_transformation_object_path , obj=one_hot_encoder)

            data_tranformation_artifact=artifact_entity.DataTransformationArtifact(
                                   transformation_object_path = self.data_transformation_config.data_transformation_object_path ,
                                   transformed_train_path =self.data_transformation_config.data_transformation_train_path,
                                   transformed_test_path =self.data_transformation_config.data_transformation_test_path
            )
            logging.info(f"Data Transformation Object {data_tranformation_artifact}")
            return data_tranformation_artifact
        
        
        except Exception as e:
            raise ProjectException(e,sys)    