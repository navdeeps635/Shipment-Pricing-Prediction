from shipment.entity import config_entity,artifact_entity
from shipment.exception import ShipmentException
from shipment.logger import logging
import os,sys
import pandas as pd
import numpy as np
from shipment import utils
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from shipment.config import TARGET_COLUMN

import warnings
warnings.filterwarnings("ignore")

class DataTransformation:

    def __init__(self,
    data_transformation_config:config_entity.DataTransformationConfig,
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise ShipmentException(e,sys)

    def get_input_transformer_object(self,numerical_columns,categorical_columns)->ColumnTransformer:
        try:
            numerical_pipeline_steps = [
                ('log_transformer',FunctionTransformer(func = np.log1p)),
                ('feature_scaler',RobustScaler())
            ]

            numerical_features_pipeline = Pipeline(numerical_pipeline_steps)

            categorical_pipeline_steps = [
                ('categorical_imputer',CategoricalImputationMICE(columns = categorical_columns)),
                ('categorical_encoder',FunctionTransformer(func = self.label_encoder)),
                ('feature_scaler',StandardScaler())
            ]

            categorical_features_pipeline = Pipeline(categorical_pipeline_steps)

            preprocessing = ColumnTransformer([
                ('numerical_features_pipeline', numerical_features_pipeline, numerical_columns),
                ('categorical_features_pipeline', categorical_features_pipeline, categorical_columns)
            ])
            
            return preprocessing
        
        except Exception as e:
                raise ShipmentException(e,sys)
    
    def get_target_transformer_object(self,)->ColumnTransformer:
        try:
            target_pipeline_steps = [
                ('log_transformer',FunctionTransformer(func = np.log1p)),
                ('feature_scaler',StandardScaler())
            ]

            target_pipeline = Pipeline(target_pipeline_steps)
            
            return target_pipeline

        except Exception as e:
            raise ShipmentException(e,sys)

    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            logging.info(f"{'>>'*10} Data Transformation {'<<'*10}")
            logging.info(f"reading train and test dataset")
            #read train and test data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"dropping unnecessary columns from train and test dataset")
            #drop unwanted columns from train and test data
            train_df = self.drop_unwanted_columns(df = train_df)
            test_df = self.drop_unwanted_columns(df = test_df)
            
            logging.info(f"cleaning text data from train and test dataset")
            #clean the columns such as remove text and fill it with NaN value
            train_df = self.clean_columns_data(df = train_df)
            test_df = self.clean_columns_data(df = test_df)

            logging.info(f"splitting input features in  train and test dataset")            
            #get numerical and categorical features from train data
            num_cols = train_df.select_dtypes(include = ['int64','float64']).columns

            numerical_imputer = NumericalImputationMICE(columns = num_cols)

            logging.info(f"imputing numerical features")
            #input features in train and test data
            train_df = numerical_imputer.fit_transform(train_df)
            test_df = numerical_imputer.transform(test_df)

            input_feature_train_df = train_df.drop(columns =[TARGET_COLUMN])
            logging.info(input_feature_train_df)         
            input_feature_test_df = test_df.drop(columns =[TARGET_COLUMN])

            logging.info(f"splitting target features in  train and test dataset")
            #target feature in train and test data 
            target_feature_train_df = train_df[TARGET_COLUMN] 
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"splitting input features in  train and test dataset")            
            #get numerical and categorical features from train data
            train_num_cols = input_feature_train_df.select_dtypes(include = ['int64','float64']).columns
            train_cat_cols = input_feature_train_df.select_dtypes(include = 'object').columns

            #creat an instance of input data transformer
            preprocessing_input_object = self.get_input_transformer_object(
                numerical_columns = train_num_cols,
                categorical_columns = train_cat_cols
            )

            logging.info(f"Transforming input features in train and test dataset")
            #transformed train and test input feuatres array
            input_feature_train_arr = preprocessing_input_object.fit_transform(input_feature_train_df)
            #print(input_feature_train_arr)
            input_feature_test_arr = preprocessing_input_object.transform(input_feature_test_df)  
            
            #print(input_feature_train_arr[:1])
            #creat an instance of target data transformer
            preprocessing_target_object = self.get_target_transformer_object()

            logging.info(f"Transforming target features in train and test dataset")
            #transformed train and test input feuatres array
            target_feature_train_arr = preprocessing_target_object.fit_transform(target_feature_train_df.values.reshape(-1,1))
            target_feature_test_arr = preprocessing_target_object.transform(target_feature_test_df.values.reshape(-1,1))

            #print(target_feature_train_arr)
            logging.info(f"Concatenating transforming train and test array")
            #conacatenate transformred input feature array and target feature array
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arr =  np.c_[input_feature_test_arr,target_feature_test_arr]
            
            logging.info(f"Saving transforming train and test array")
            #save numpy array
            utils.save_numpy_array_data(file_path = self.data_transformation_config.tranformed_train_path, array = train_arr)
            utils.save_numpy_array_data(file_path = self.data_transformation_config.tranformed_test_path, array = test_arr)

            logging.info(f"Saving input and target transformer pkl file")
            #save transofrmed data into pkl file
            utils.save_object(file_path = self.data_transformation_config.numerical_imputer_object_path, object = numerical_imputer)
            utils.save_object(file_path = self.data_transformation_config.input_transformer_object_path, object = preprocessing_input_object)
            utils.save_object(file_path = self.data_transformation_config.target_transformer_object_path, object = preprocessing_target_object)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                tranformed_train_path = self.data_transformation_config.tranformed_train_path,
                tranformed_test_path = self.data_transformation_config.tranformed_test_path,
                numerical_imputer_object_path = self.data_transformation_config.numerical_imputer_object_path,
                input_transformer_object_path = self.data_transformation_config.input_transformer_object_path,
                target_transformer_object_path = self.data_transformation_config.target_transformer_object_path
            )
            
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")

            return data_transformation_artifact

        except Exception as e:
            raise ShipmentException(e,sys)

    #function to drop unwanted columns
    def drop_unwanted_columns(self,df)->pd.DataFrame:
        try:
            #drop unnecessary columns as they are not required for model training
            columns_to_drop =  ['ID', 'Project Code', 'PQ #', 'PO / SO #', 'ASN/DN #','Item Description','PQ First Sent to Client Date',
            'PO Sent to Vendor Date','Scheduled Delivery Date','Delivered to Client Date','Delivery Recorded Date']

            df = df.drop(columns=columns_to_drop)
            
            return df

        except Exception as e:
                raise ShipmentException(e,sys)

    #function to encode top 20 categories from every catrgorical featutre while rest are considered zero
    def label_encoder(self,X,y = None):        
        try:
            for col in X:
                top_categories_locations = X.loc[:,col].value_counts()[:20].index
                other_locations = X.loc[:,col].value_counts()[20:].index

                labels = {}
                for label,location in enumerate(top_categories_locations):
                    labels[location] = label

                for location in other_locations:
                    labels[location] = 0
                
                X.loc[:,col] = X.loc[:,col].map(labels)

            return X

        except Exception as e:
                raise ShipmentException(e,sys)

    #create target feature
    def create_target_feature(self,df):
        try:
            new_df = df.copy()
            new_df['Shipment Price'] = new_df['Line Item Value'] + new_df['Freight Cost (USD)'] + new_df['Line Item Insurance (USD)']

            return new_df['Shipment Price']
        
        except Exception as e:
                raise ShipmentException(e,sys)
    
    #clean column data
    def clean_columns_data(self,df):
        try:
            #create a copy of dataframe
            new_df = df.copy()

            #replace text starting with "See" and "Invoiced" with np.nan in "Freight Cost" feature 
            new_df.loc[(new_df['Freight Cost (USD)'].str.contains('See')) | (new_df['Freight Cost (USD)'].str.contains('Invoiced')),'Freight Cost (USD)'] = np.nan
            
            #replace text "Freight Included in Commodity Cost" with 0 in "Freight Cost" feature
            new_df.loc[new_df['Freight Cost (USD)'].str.contains('Commodity', na = False),'Freight Cost (USD)'] = 0

            #replace text starting with "See" and "Captured" with np.nan in "Weight" feature
            new_df.loc[(new_df['Weight (Kilograms)'].str.contains('See')) | (new_df['Weight (Kilograms)'].str.contains('Captured')),'Weight (Kilograms)'] = np.nan

            #convert datatype of 'Freight Cost' and 'Weight' to float.
            new_df['Freight Cost (USD)'] = new_df['Freight Cost (USD)'].astype('float')
            new_df['Weight (Kilograms)'] = new_df['Weight (Kilograms)'].astype('float')

            return new_df

        except Exception as e:
                raise ShipmentException(e,sys)

#create a class to make custom tranformer for numerical features imputation
class NumericalImputationMICE(BaseEstimator, TransformerMixin):
    
    def __init__(self,columns):
        try:
            self.columns = columns
            return None
        except Exception as e:
            raise ShipmentException(e,sys)
    
    def fit(self, X, y = None):
        #the type of X might be a DataFrame or a NumPy array depending on the previous transformer object that you use in the pipeline
        return self
    
    def transform(self, X, y = None):
        """
        Impute numeric data using MICE imputation with Decision Tree Regressor.
        (we can use any other regressors to impute the data)
        """
        try:
            impute_numeric = IterativeImputer(estimator = DecisionTreeRegressor(),max_iter = 3,initial_strategy = "mean")
            imputed_data = impute_numeric.fit_transform(X[self.columns])
            X[self.columns] =  imputed_data.astype(int)

            return X

        except Exception as e:
            raise ShipmentException(e,sys)

#create a class to make custom tranformer for categorical features imputation
class CategoricalImputationMICE(BaseEstimator, TransformerMixin): 
    
    def __init__(self,columns):
        try:
            self.columns = columns
            return None

        except Exception as e:
            raise ShipmentException(e,sys)

    def fit(self, X, y = None):
        #the type of X might be a DataFrame or a NumPy array depending on the previous transformer object that you use in the pipeline

        return self
    
    def transform(self, X, y = None):
        """
        Impute categoric data using MICE imputation with Decision Tree Classifier.
        Steps:
        1. Ordinal Encode the non-null values
        2. Use MICE imputation with Decision Tree Classifier to impute the ordinal encoded data
        (we can use any other classifier to impute the data)
        3. Inverse transform the ordinal encoded data.
        """
        try:
            fit_encoder={}
            for col in self.columns:
                
                #Label encode train data
                nn_vals = X[col][X[col].notnull()]
                fit_encoder[col] = LabelEncoder().fit(nn_vals)
                nn_vals_arr = np.array(fit_encoder[col].transform(nn_vals)).reshape(-1,)
                X[col].loc[X[col].notnull()] = nn_vals_arr

            #Impute the data using MICE with Gradient Boosting Classifier
            impute_categoric = IterativeImputer(estimator = DecisionTreeClassifier(), max_iter = 3, initial_strategy='most_frequent')
            imputed_data = impute_categoric.fit_transform(X[self.columns])

            X[self.columns] =  imputed_data.astype(int)

            #Inverse Transform categorical features
            for col in self.columns:
                X[col] = fit_encoder[col].inverse_transform(X[col])

            return X[self.columns]

        except Exception as e:
            raise ShipmentException(e,sys)