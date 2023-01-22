from shipment.pipeline.training_pipeline import  start_training_pipeline
import os,sys
from shipment.exception import ShipmentException

if __name__ =='__main__':
    try:
        start_training_pipeline()

    except Exception as e:
        ShipmentException(e,sys)