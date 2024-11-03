from dataclasses import dataclass 


@dataclass

class DataIngestionArtifact:

    feature_store_file_path :str
    train_file_path:str
    test_file_path:str

@dataclass 
class DataValidationArtifact:
    report_file_path:str   


@dataclass
class DataTransformationArtifact:
    transformation_object_path:str    
    transformed_train_path:str
    transformed_test_path:str

@dataclass
class ModelTrainerArtifact:
    model_path:str
    f1_train_score:str
    f1_test_score:str
    
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float    