from shipment.entity import config_entity,artifact_entity,model_finder
from shipment.exception import ShipmentException
from shipment.logger import logging
import os,sys
import pandas as pd
import numpy as np
from shipment import utils

class ModelTrainer:

    def __init__(self,
    model_trainer_config:config_entity.ModelTrainerConfig,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    best_model:model_finder.SelectBestModel):
        
        try:
            logging.info(f"{'>>'*20} Model Trainer Initiated {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.best_model = best_model

        except Exception as e:
            raise ShipmentException(e, sys)
    
    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.tranformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.tranformed_test_path)
            
            logging.info(f"split input and target feature from train and test array")
            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info(f"Best trained model and corresponding training_r2_score")
            #get best model and corresponding r2 score for training data
            model,train_r2_score, test_r2_score = self.best_model.get_best_model(
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test
            )

            logging.info(f"Best model: {train_model}, training_r2_score: {train_r2_score} and test_r2_score: {test_r2_score}")

            #check for expected score
            logging.info(f"check if model is underfitted or not")
            if test_r2_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give\
                    expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {test_r2_score}")

            #check for overfitting or underfitting
            logging.info(f"check if model is overfitted or not")
            diff = abs(train_r2_score - test_r2_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score difference is more than overfitting threshold: {self.model_trainer_config.overfitting_threshold * 100}%")

            #save the training model
            logging.info(f"save model object")
            utils.save_object(file_path = self.model_trainer_config.model_path, object = train_model)

            #prepare artifact
            logging.info(f"prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path = self.model_trainer_config.model_path,
                train_r2_score = train_r2_score,
                test_r2_score = test_r2_score
            )
            logging.info(f"{'>>'*20} Model Trainer Completed {'<<'*20}")

            return model_trainer_artifact
        
        except Exception as e:
            raise ShipmentException(e, sys)

            