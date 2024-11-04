from src.pipeline.trainng_pipeline import start_training_pipeline
from src.pipeline.batch_prediction import start_batch_prediction
from src.exception import ProjectException
import os , sys

file_path = os.path.join(os.getcwd() , "dataset/HRDataset_v14.csv")


if __name__=="__main__":

    try:
       training_ouput=start_training_pipeline()
       batch_output=start_batch_prediction(file_path)
       print(">"*15," Current Prediction is " , ">"*15)
       print(batch_output)
       
    except Exception as e:
        raise ProjectException(e ,sys)   