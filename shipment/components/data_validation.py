import os,sys
from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.entity import config_entity,artifact_entity
from scipy.stats import ks_2samp
from shipment import utils
import pandas as pd

class DataValidation:

    def __init__(self,
        data_validation_config:config_entity.DataValidationConfig,
        data_ingstion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingstion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise ShipmentException(e,sys)

    def drop_missing_columns(self,df,report_key_name:str):

        '''
        This function will drop the columns having missing vlaues more than threshold

        Params:
        df: Accepts a pandas dataframe
        report_key_name: keyname for tracking the steps
        =================================================
        returns Pandas DataFrame if atleast a single column is available after dropping columns else None
        '''
        
        try:
            threshold = self.data_validation_config.missing_threshold

            missing_values_report = df.isnull().sum()/df.shape[0]

            drop_column_names = missing_values_report[missing_values_report>threshold].index

            self.validation_error[report_key_name] = list(drop_column_names)

            df.drop(columns = drop_column_names,inplace = True)

            if len(df.columns) == 0:
                return None
            
            return df
        
        except Exception as e:
            raise ShipmentException(e,sys)
        
    def is_required_column_exist(self,base_df,current_df,report_key_name:str)-> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []

            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name] = missing_columns
                return False

            return True
        
        except Exception as e:
            raise ShipmentException(e,sys)    

    def data_drift(self,base_df,current_df,report_key_name:str):
        try:
            drift_report = []
            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_columns],current_df[base_columns]

                response = ks_2samp(data1 = base_data, data2 = current_data)

                if response.pvalue>0.05:
                    #null hypothesis accepted
                    drift_report[base_column] = {"Pvalue":float(response.pvalue),"same distribution":True}
                else:
                    #null hypothesis rejected
                    drift_report[base_column] = {"Pvalue":float(response.pvalue),"same distribution":False}
            
            self.validation_error[report_key_name] = drift_report
        
        except Exception as e:
            raise ShipmentException(e,sys)

    def initiate_data_validation(self,)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            logging.info(f"drop null values columns from base dataframe")
            #drop missing values columns from base_df
            base_df = self.drop_missing_columns(df= base_df, report_key_name = 'missing_values_within_base_dataset')

            logging.info(f"reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            
            logging.info(f"reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"drop null values columns from training dataframe")
            #drop missing values columns from train_df
            train_df = self.drop_missing_columns(df = train_df, report_key_name = 'missing_values_within_train_dataset')
            
            logging.info(f"drop null values columns from test dataframe")
            #drop missing values columns from train_df
            test_df = self.drop_missing_columns(df = test_df, report_key_name = 'missing_values_within_test_dataset')

            logging.info(f"is all required columns present in training dataframe")
            train_df_columns_status = self.is_required_column_exist(
                base_df = base_df, current_df = train_df, report_key_name = 'missing_columns_within_train_dataset')


            logging.info(f"is all required columns present in test dataframe")
            test_df_columns_status = self.is_required_column_exist(
                base_df = base_df, current_df = test_df, report_key_name = 'missing_columns_within_test_dataset')

            if train_df_columns_status:
                logging.info(f"As all columns are available in training dataframe hence detecting data drift")
                self.data_drift(base_df = base_df, current_df = train_df, report_key_name="data_drift_within_train_dataset")
            
            if test_df_columns_status:
                logging.info(f"As all columns are available in test dataframe hence detecting data drift")
                self.data_drift(base_df = base_df, current_df = test_df, report_key_name="data_drift_within_test_dataset")
            
            #write the report
            logging.info(f"writing report in yaml file")
            utils.write_yaml_file(file_path = self.data_validation_config.report_file_path, data = self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path = self.data_validation_config.report_file_path)

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        
        except Exception as e:
            raise ShipmentException(e,sys)