from shipment.exception import ShipmentException
from shipment.logger import logging
import sys

try:
    logging.info("Project Started")
    3/0
except Exception as e:
    raise ShipmentException(e,sys)