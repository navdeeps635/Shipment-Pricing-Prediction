from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.entity import config_entity,artifact_entity
from shipment.components.data_ingestion import DataIngestion
import os,sys

if __name__ == '__main__':
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config)

        data_ingestion = DataIngestion(data_ingestion_config = data_ingestion_config)

        data_ingstion_artifact = data_ingestion.initiate_data_ingestion()

    except Exception as e:
        raise ShipmentException(e,sys)