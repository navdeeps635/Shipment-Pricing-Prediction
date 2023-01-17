from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str

@dataclass
class DataTransformationArtifact:
    tranformed_train_path:str
    tranformed_test_path:str
    input_transformer_object_path:str
    target_transformer_object_path:str

@dataclass
class ModelTrainerArtifact:
    model_path:str
    train_r2_score:float
    test_r2_score:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_score:float
    
@dataclass
class ModelPusherArtifact:...