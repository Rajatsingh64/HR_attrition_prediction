from src.entity import artifact_entity, config_entity
from src.exception import ProjectException
from src.predictor import ModelResolver
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from src.config import TARGET_COLUMN, important_features
from src.utils import parse_date_DOB, calculate_age, calculate_tenure, parse_date_for_tenure, load_object
import warnings
warnings.filterwarnings("ignore")

class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        """
        Initialize ModelEvaluation class with configuration and artifacts.
        """
        try:
            logging.info(f'{"="*20} Starting Model Evaluation {"="*20}')
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        """
        Compares the current model with the previously saved production model and evaluates performance.
        Returns:
            ModelEvaluationArtifact: Contains evaluation result & improved accuracy.
        """
        try:
            logging.info("Checking if a previously saved model exists for comparison...")
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            # If no previously saved model exists, accept the current model by default
            if latest_dir_path is None:
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None
                )
                logging.info(f"No previous model found. Accepting current model by default. Artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            # Load previous transformer & model
            logging.info("Loading previously saved transformer and model...")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()

            logging.info(f"Previous model path: {model_path}")
            logging.info(f"Previous transformer path: {transformer_path}")

            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)

            # Load current trained transformer & model
            logging.info("Loading current trained transformer and model...")
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_transformer = load_object(file_path=self.data_transformation_artifact.transformation_object_path)

            # Load test dataset
            logging.info("Loading test dataset...")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Drop unnecessary columns (only keep important features)
            not_important_features = [col for col in test_df.columns if col not in important_features]
            test_df.drop(not_important_features, axis=1, inplace=True)
            logging.info(f"Dropped unimportant columns: {not_important_features}")

            # Preprocessing steps
            logging.info("Preprocessing test dataset: Calculating Age and Tenure...")
            test_df["DOB"] = test_df["DOB"].apply(parse_date_DOB)
            test_df["Age"] = test_df["DOB"].apply(calculate_age)

            # Replace unrealistic ages (>50) with mean age
            mean_age = test_df['Age'][test_df['Age'] <= 50].mean()
            test_df['Age'] = test_df['Age'].apply(lambda x: mean_age if x > 50 else x)

            # Calculate Tenure
            test_df['DateofHire'] = test_df['DateofHire'].apply(parse_date_for_tenure)
            test_df['DateofTermination'] = test_df['DateofTermination'].apply(parse_date_for_tenure)
            test_df['Tenure'] = test_df.apply(calculate_tenure, axis=1)
            test_df.drop(["DOB", "DateofHire", "DateofTermination"], axis=1, inplace=True)

            # Split input & target
            input_test_df = test_df.drop(TARGET_COLUMN, axis=1)
            target_df = test_df[TARGET_COLUMN]

            # ===============================
            # Previous Model Evaluation
            # ===============================
            logging.info("Evaluating Previous Production Model...")
            input_features_name = list(transformer.feature_names_in_)
            test_encoded = transformer.transform(input_test_df[input_features_name])
            test_df_encoded = pd.DataFrame(test_encoded, columns=transformer.get_feature_names_out(input_features_name))
            input_df = pd.concat(
                [input_test_df.drop(columns=input_features_name).reset_index(drop=True),
                 test_df_encoded.reset_index(drop=True)],
                axis=1
            )

            y_pred = model.predict(input_df)
            previous_model_score = f1_score(target_df, y_pred)
            logging.info(f"Previous Model F1 Score: {previous_model_score:.4f}")

            # ===============================
            # Current Model Evaluation
            # ===============================
            logging.info("Evaluating Current Trained Model...")
            input_feature_name = list(current_transformer.feature_names_in_)
            input_encoded_features = current_transformer.transform(input_test_df[input_feature_name])
            test_df_encoded = pd.DataFrame(input_encoded_features, columns=current_transformer.get_feature_names_out(input_feature_name))
            input_df = pd.concat(
                [input_test_df.drop(columns=input_feature_name).reset_index(drop=True),
                 test_df_encoded.reset_index(drop=True)],
                axis=1
            )

            y_pred = current_model.predict(input_df)
            current_model_score = f1_score(target_df, y_pred)
            logging.info(f"Current Model F1 Score: {current_model_score:.4f}")

            # ===============================
            # Comparison and Acceptance
            # ===============================
            if current_model_score <= previous_model_score:
                logging.info("Current model did NOT outperform the previous model. Rejecting current model.")
                raise Exception("Current trained model is not better than previous model.")

            improvement = current_model_score - previous_model_score
            logging.info(f"Current model outperformed previous model by {improvement:.4f} points.")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=improvement
            )
            logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise ProjectException(e, sys)
