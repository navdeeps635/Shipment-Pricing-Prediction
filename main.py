from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.entity import config_entity,artifact_entity,model_finder
from shipment.components.data_ingestion import DataIngestion
#from shipment.components.data_validation import DataValidation
from shipment.components.data_transformation import DataTransformation
from shipment.components.model_trainer import ModelTrainer

import os,sys

if __name__ == '__main__':
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config = data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # #data validation
        # data_validation_config = config_entity.DataValidationConfig(training_pipeline_config)
        # data_validation = DataValidation(
        #     data_validation_config = data_validation_config,
        #     data_ingstion_artifact = data_ingstion_artifact)
        # data_validation_artifact = data_validation.initiate_data_validation()

        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_transformation_config = data_transformation_config,
            data_ingestion_artifact = data_ingestion_artifact
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config = model_trainer_config,
            data_transformation_artifact = data_transformation_artifact,
            best_model = model_finder.SelectBestModel()
        )

        model_trainer_artifact = model_trainer.initiate_model_trainer()
    except Exception as e:
        raise ShipmentException(e,sys)