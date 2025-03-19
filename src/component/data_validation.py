from src.entity import artifact_entity, config_entity
from src.exception import ProjectException
from src.logger import logging
from scipy.stats import ks_2samp, chi2_contingency
from typing import Optional
import os, sys
import pandas as pd
from src import utils
import numpy as np
from src.config import TARGET_COLUMN

class DataValidation:

    def __init__(self, data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation Started {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise ProjectException(e, sys)

    def drop_missing_values_columns(self, df: pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum() / df.shape[0]
            drop_column_names = null_report[null_report > threshold].index

            logging.info(f"Dropping columns with null values above threshold  during validation{threshold}: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names), axis=1, inplace=True)

            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise ProjectException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = [col for col in base_columns if col not in current_columns]

            if missing_columns:
                logging.warning(f"Missing columns: {missing_columns}")
                self.validation_error[report_key_name] = missing_columns
                return False
            return True
        except Exception as e:
            raise ProjectException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        try:
            drift_report = dict()
            base_columns = base_df.columns

            for base_column in base_columns:
                if base_column == TARGET_COLUMN:
                    continue  # Skip target column for drift detection

                base_data = base_df[base_column]
                current_data = current_df[base_column]

                # Numerical
                if base_data.dtype in [np.float64, np.int64]:
                    ks_test = ks_2samp(base_data, current_data)
                    drift_report[base_column] = {
                        "p_value": float(ks_test.pvalue),
                        "same_distribution": bool(ks_test.pvalue > 0.05)
                    }

                # Categorical
                elif base_data.dtype == 'object':
                    combined = pd.concat([base_data, current_data], axis=1).dropna()
                    if combined.shape[0] == 0:
                        drift_report[base_column] = {
                            "p_value": None,
                            "same_distribution": None,
                            "note": "No common non-null values to compare."
                        }
                        continue
                    contingency_table = pd.crosstab(combined.iloc[:, 0], combined.iloc[:, 1])
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        # Not enough categories for chi2
                        drift_report[base_column] = {
                            "p_value": None,
                            "same_distribution": None,
                            "note": "Insufficient categories to perform chi-square test."
                        }
                        continue
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    drift_report[base_column] = {
                        "p_value": float(p_value),
                        "same_distribution": bool(p_value > 0.05)
                    }

            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            # Read base dataset
            logging.info(f"Reading base dataframe: {self.data_validation_config.base_file_path}")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na": np.NAN}, inplace=True)
            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within_base_dataset")
            if base_df is None:
                raise ProjectException("No columns left in base dataset after removing null columns.", sys)

            # Read train & test
            logging.info(f"Reading train dataframe: {self.data_ingestion_artifact.train_file_path}")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)

            logging.info(f"Reading test dataframe: {self.data_ingestion_artifact.test_file_path}")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_within_train_dataset")
            test_df = self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_test_dataset")

            # Convert numerical columns
            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            # Check required columns
            logging.info(f"Checking required columns in train dataset")
            train_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df,
                                                           report_key_name="missing_columns_within_train_dataset")
            logging.info(f"Checking required columns in test dataset")
            test_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df,
                                                          report_key_name="missing_columns_within_test_dataset")

            # Stop if required columns missing
            if not train_status or not test_status:
                raise ProjectException("Missing required columns in training or test dataset.", sys)

            # Data drift detection
            logging.info(f"Detecting data drift in train dataset")
            self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")

            logging.info(f"Detecting data drift in test dataset")
            self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")

            # Write report
            logging.info(f"Writing validation report to {self.data_validation_config.report_file_path}")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                  data=self.validation_error)

            logging.info(f"Validation report saved successfully.")

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path
            )
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")

            logging.info(f"{'>>'*20} Data Validation Completed {'<<'*20}")

            return data_validation_artifact

        except Exception as e:
            raise ProjectException(e, sys)
