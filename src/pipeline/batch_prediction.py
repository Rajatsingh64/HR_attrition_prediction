from src.exception import ProjectException
from src.logger import logging
from src.predictor import ModelResolver
import pandas as pd
from src.utils import load_object
import os,sys
from datetime import datetime
from src.config import important_features ,TARGET_COLUMN
from src.utils import parse_date_DOB , parse_date_for_tenure , calculate_age , calculate_tenure
PREDICTION_DIR="prediction"

import numpy as np
def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace=True)
        #validation
        not_important_features = [ column for column in df.columns if column not in important_features]
        df.drop(not_important_features, axis=1 , inplace=True) #Lets drop unnecessary column
        df["DOB"] = df["DOB"].apply(parse_date_DOB)
        df['Age'] = df['DOB'].apply(calculate_age)
        mean_age = df['Age'][df['Age'] <= 50].mean()
        df['Age'] = df['Age'].apply(lambda x: mean_age if x > 50 else x)

        df['DateofHire'] = df['DateofHire'].apply(parse_date_for_tenure)
        df['DateofTermination'] = df['DateofTermination'].apply(parse_date_for_tenure)
        df['Tenure'] = df.apply(calculate_tenure, axis=1)
        df.drop(["DOB" , "DateofHire" ,"DateofTermination"] ,axis=1 ,inplace=True)
        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        feature_encoded = transformer.transform(df[input_feature_names])
        feature_encoded_df=pd.DataFrame(feature_encoded, columns=transformer.get_feature_names_out(input_feature_names))
        df_encoded=pd.concat([df.drop(columns=input_feature_names).reset_index(drop=True), feature_encoded_df.reset_index(drop=True)], axis=1)
        input_df=df_encoded.drop(TARGET_COLUMN , axis=1)
        target_df=df_encoded[TARGET_COLUMN] #In my case my Target is already Encoded 
        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_df)
        

        cat_prediction = target_encoder.inverse_transform(prediction)

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise ProjectException(e, sys)