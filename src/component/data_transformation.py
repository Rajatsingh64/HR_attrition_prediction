from src.entity import artifact_entity, config_entity
from src.exception import ProjectException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os, sys
from src.config import TARGET_COLUMN, important_features, nominal_features
from src.utils import (
    calculate_age,
    calculate_tenure,
    parse_date_DOB,
    parse_date_for_tenure,
    save_numpy_array_data,
    save_object,
)
import warnings
warnings.filterwarnings("ignore")

class DataTransformation:
    def __init__(
        self,
        data_transformation_config: config_entity.DataTransformationConfig,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
    ):
        try:
            logging.info(f'{">"*20} Data Transformation {"<"*20}')
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            logging.info("Reading Train and Test data as DataFrame")
            # Reading datasets
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Dropping unnecessary columns
            not_important_features = [
                column for column in train_df.columns if column not in important_features
            ]
            train_df.drop(not_important_features, axis=1, inplace=True, errors="ignore")
            test_df.drop(not_important_features, axis=1, inplace=True, errors="ignore")
            logging.info(f"Dropped unnecessary columns")

            # Parsing DOB
            logging.info("Converting DOB to pd.datetime format")
            train_df["DOB"] = train_df["DOB"].apply(parse_date_DOB)
            test_df["DOB"] = test_df["DOB"].apply(parse_date_DOB)

            # Calculate Age
            logging.info("Calculating Age")
            train_df["Age"] = train_df["DOB"].apply(calculate_age)
            test_df["Age"] = test_df["DOB"].apply(calculate_age)

            # Replace age > threshold
            AGE_THRESHOLD = 50
            mean_age = train_df["Age"][train_df["Age"] <= AGE_THRESHOLD].mean()
            train_df["Age"] = train_df["Age"].apply(lambda x: mean_age if x > AGE_THRESHOLD else x)
            test_df["Age"] = test_df["Age"].apply(lambda x: mean_age if x > AGE_THRESHOLD else x)
            logging.info(f"Replaced Age > {AGE_THRESHOLD} with average")

            # Calculate Tenure
            logging.info("Calculating Tenure")
            train_df["DateofHire"] = train_df["DateofHire"].apply(parse_date_for_tenure)
            train_df["DateofTermination"] = train_df["DateofTermination"].apply(parse_date_for_tenure)
            train_df["Tenure"] = train_df.apply(calculate_tenure, axis=1)

            test_df["DateofHire"] = test_df["DateofHire"].apply(parse_date_for_tenure)
            test_df["DateofTermination"] = test_df["DateofTermination"].apply(parse_date_for_tenure)
            test_df["Tenure"] = test_df.apply(calculate_tenure, axis=1)

            # Drop DOB and Hire/Termination columns
            train_df.drop(["DOB", "DateofHire", "DateofTermination"], axis=1, inplace=True)
            test_df.drop(["DOB", "DateofHire", "DateofTermination"], axis=1, inplace=True)
            logging.info("Dropped DOB, DateofHire, DateofTermination columns")

            # One-Hot Encoding
            logging.info(f"Starting OneHotEncoding for: {nominal_features}")
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

            train_encoded_features = one_hot_encoder.fit_transform(train_df[nominal_features])
            train_encoded_df = pd.DataFrame(
                train_encoded_features, columns=one_hot_encoder.get_feature_names_out(nominal_features)
            )

            test_encoded_features = one_hot_encoder.transform(test_df[nominal_features])
            test_encoded_df = pd.DataFrame(
                test_encoded_features, columns=one_hot_encoder.get_feature_names_out(nominal_features)
            )
            logging.info(f"Completed OneHotEncoding for: {nominal_features}")

            # Concatenate encoded features
            train_df = pd.concat(
                [train_df.drop(columns=nominal_features).reset_index(drop=True), train_encoded_df.reset_index(drop=True)],
                axis=1,
            )
            test_df = pd.concat(
                [test_df.drop(columns=nominal_features).reset_index(drop=True), test_encoded_df.reset_index(drop=True)],
                axis=1,
            )

            # Split features & target
            input_features_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_features_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Separated input and target features")

            # Apply SMOTE only on training data
            smote = SMOTE()
            logging.info(
                f"Before SMOTE - Train input: {input_features_train_df.shape}, Target: {target_feature_train_df.shape}"
            )
            input_features_train_df, target_feature_train_df = smote.fit_resample(
                input_features_train_df, target_feature_train_df
            )
            logging.info(
                f"After SMOTE - Train input: {input_features_train_df.shape}, Target: {target_feature_train_df.shape}"
            )

            # Convert to numpy arrays
            train_arr = np.c_[input_features_train_df, target_feature_train_df]
            test_arr = np.c_[input_features_test_df, target_feature_test_df]

            # Save numpy arrays and encoder
            save_numpy_array_data(
                file_path=self.data_transformation_config.data_transformation_train_path, array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.data_transformation_test_path, array=test_arr
            )
            save_object(self.data_transformation_config.data_transformation_object_path, obj=one_hot_encoder)

            # Prepare artifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformation_object_path=self.data_transformation_config.data_transformation_object_path,
                transformed_train_path=self.data_transformation_config.data_transformation_train_path,
                transformed_test_path=self.data_transformation_config.data_transformation_test_path,
            )
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise ProjectException(e, sys)
