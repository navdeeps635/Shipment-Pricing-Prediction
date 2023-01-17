from shipment.entity import config_entity,artifact_entity,model_finder
from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.predictor import ModelResolver
from shipment.data_transformation import DataTransformation
from sklearn.metrics import r2_score
import os,sys
import pandas as pd

class ModelEvaluation:

    def __init__(self,
    data_ingestion_artifact:artifact_entity.data_ingestion_artifact,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact,
    model_eval_config:config_entity.ModelEvaluationConfig   
    ):

        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact,
            self.data_transformation_artifact = data_transformation_artifact,
            self.model_trainer_artifact = model_trainer_artifact,
            self.model_eval_config = model_eval_config,
            self.model_resolver = ModelResolver()
            self.data_transformer = DataTransformation()

        except Exception as e:
            ShipmentException(e,sys)

    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:

        try:
            logging.info("if saved model folder has model the we will compare")
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted = True, improved_score = None
                )
            
                logging.info(f"Model Evaluation Artifact:{model_eval_artifact}")
                return model_eval_artifact

            #Finding location of input transformer, target transformer and model
            logging.info(f"finding location of input transformer, target transformer and model")
            model_path = self.model_resoler.get_latest_model_path()
            input_transformer_object_path = self.get_latest_input_transformer_path()
            target_transformer_object_path = self.get_latest_target_transformer_path()

            logging.info(f"Previous trained objects of transformer, model and target encoder")
            #load previously trained objects
            model = load_object(file_path = model_path)
            input_transformer = load_object(file_path = input_transformer_object_path)
            target_transformer = load_object(file_path = target_transformer_object_path)

            logging.info(f"Currently trained model objects")
            #currently trained model objects
            current_model = load_object(file_path = self.model_trainer_artifact.model_path)
            current_input_transformer = load_object(file_path = self.data_transformation_artifact.input_transformer_object_path)
            current_target_transformer = load_object(file_path = self.data_transformation_artifact.target_transformer_object_path)

            #get test dataset
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            logging.info(f"Cleaning input data")
            #remove unwanted columns and clean the columns in input_df 
            input_df = self.data_transformer.drop_unwanted_columns(df = test_df)
            input_df = self.data_transformer.clean_columns_data(df = input_df)

            logging.info(f"Creating target feature")
            #create target feature
            target_df = self.data_transformer.create_target_feature(df = input_df)
            
            logging.info(f"Making Prediction based on previous model")
            #score using previous trained model
            input_feautres_name = list(input_transformer.feature_names_in_)
            input_arr = input_transformer.transform(input_df[input_feautres_name])
            y_pred = model.predict(input_arr)
            y_true = target_transformer.transform(target_df)
            print(f"Prediction using previous model:{target_transformer.inverse_transform(y_pred[:5])}")

            previous_model_score = r2_score(y_true = y_true, y_pred = y_pred)
            logging.info(f"Score using previous trained model: {previous_model_score}")

            logging.info(f"Making Prediction based on currently trained model")
            #score using currently trained model
            input_feautres_name = list(input_transformer.feature_names_in_)
            current_input_arr = current_input_transformer.transform(input_df[input_feautres_name])
            current_y_pred = current_model.predict(input_arr)
            y_true = current_target_transformer.transform(target_df)
            print(f"Prediction using previous model:{current_target_transformer.inverse_transform(y_pred[:5])}")

            current_model_score = r2_score(y_true = y_true, y_pred = current_y_pred)
            logging.info(f"Score using previous trained model: {current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted = True,
                improved_score = current_model_score - previous_model_score
            )            

            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise ShipmentException(e,sys)