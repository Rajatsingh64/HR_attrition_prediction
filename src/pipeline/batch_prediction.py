from src.exception import ProjectException
from src.logger import logging
from src.predictor import ModelResolver
import pandas as pd
import numpy as np
from src.utils import load_object
import os, sys
from datetime import datetime
from src.config import important_features, TARGET_COLUMN
from src.utils import parse_date_DOB, parse_date_for_tenure, calculate_age, calculate_tenure
import warnings
warnings.filterwarnings("ignore")

PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    try:
        logging.info(f'{">"*20} Batch Prediction Started {"<"*20}')
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        
        logging.info(f"Creating ModelResolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        
        logging.info(f"Reading input file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na": np.NAN}, inplace=True)
        
        # Drop unnecessary columns not in important_features
        not_important_features = [column for column in df.columns if column not in important_features]
        df.drop(not_important_features, axis=1, inplace=True)
        
        # Parsing date of birth and calculating age
        df["DOB"] = df["DOB"].apply(parse_date_DOB)
        df['Age'] = df['DOB'].apply(calculate_age)
        mean_age = df['Age'][df['Age'] <= 50].mean()
        df['Age'] = df['Age'].apply(lambda x: mean_age if x > 50 else x)
        
        # Parsing hire and termination dates, calculating tenure
        df['DateofHire'] = df['DateofHire'].apply(parse_date_for_tenure)
        df['DateofTermination'] = df['DateofTermination'].apply(parse_date_for_tenure)
        df['Tenure'] = df.apply(calculate_tenure, axis=1)
        df.drop(["DOB", "DateofHire", "DateofTermination"], axis=1, inplace=True)
        
        logging.info(f'{">"*20} Age and Tenure Features Calculated {"<"*20}')
        logging.info(f"Loading transformer to transform dataset")
        
        # Load transformer object
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        input_feature_names = list(transformer.feature_names_in_)
        
        # Transform input features
        transformed_features = transformer.transform(df[input_feature_names])
        transformed_df = pd.DataFrame(
            transformed_features, 
            columns=transformer.get_feature_names_out(input_feature_names)
        )
        
        # Merge transformed features with remaining columns
        df_encoded = pd.concat(
            [df.drop(columns=input_feature_names).reset_index(drop=True), 
             transformed_df.reset_index(drop=True)], axis=1
        )
        
        logging.info(f'{">"*20} Selecting Input Features {"<"*20}')
        
        # Prepare input and target columns
        input_df = df_encoded.drop(TARGET_COLUMN, axis=1)
        target_df = df_encoded[TARGET_COLUMN]
        
        # Load latest model
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_df)
        
        # Add prediction columns
        df["Prediction"] = prediction
        df["Cat_Prediction"] = df["Prediction"].replace({0: "Not Terminated", 1: "Terminated"})
        
        # Save prediction output
        prediction_file_name = os.path.basename(input_file_path).replace(
            ".csv", f"__{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"
        )
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)
        
        logging.info(f'{">"*20} Batch Prediction Completed Successfully {"<"*20}')
        return prediction_file_path

    except Exception as e:
        raise ProjectException(e, sys)
