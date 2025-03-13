from src.pipeline.batch_prediction import start_batch_prediction
import os ,sys

file_path=os.path.join(os.getcwd() , "dataset\HRDataset_v14.csv")
start_batch_prediction(input_file_path=file_path)
print("batch prediction completed sucessfully")