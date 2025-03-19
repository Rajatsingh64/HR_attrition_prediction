from dataclasses import dataclass
from typing import Optional

# Artifact for Data Ingestion stage
@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str

# Artifact for Data Validation stage
@dataclass
class DataValidationArtifact:
    report_file_path: str

# Artifact for Data Transformation stage
@dataclass
class DataTransformationArtifact:
    transformation_object_path: str
    transformed_train_path: str
    transformed_test_path: str

# Artifact for Model Training stage
@dataclass
class ModelTrainerArtifact:
    model_path: str
    f1_train_score: float
    f1_test_score: float

# Artifact for Model Evaluation stage
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: Optional[float] = None  # Optional in case no previous model exists for comparison

# Artifact for Model Pusher stage
@dataclass
class ModelPusherArtifact:
    pusher_model_dir: str
    saved_model_dir: str
