from shipment.logger import logging
from shipment.exception import ShipmentException
from shipment.entity import config_entity,artifact_entity
import os,sys
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json

class DataValidation:

    def __init__(self,
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
    data_validation_artifact:artifact_entity.DataValidationArtifact
    ):
        try:
            logging.info(f"{'>>'*20} Data Validation Initiated {'<<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
        except Exception as e:
            raise ShipmentException(e,sys)    
    
    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df,test_df

        except Exception as e:
            raise ShipmentException(e,sys)

    def is_train_test_file_exists(self)->bool:
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")

            if not is_available:
                training_file = self.data_ingstion_artifact.train_file_path
                test_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file} is not present"
                raise Exception(message)

            return is_available
        
        except Exception as e:
            raise ShipmentException(e,sys)
    
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False
            
            #assigment validate training and testing dataset using schema file

            

            