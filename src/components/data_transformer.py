import os
import sys
path = "/home/er_bim/Car-Price-Prediction/"
sys.path.append(path)
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"data_preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation before model training
        
        '''
        try:
            numerical_columns = ['age',
                            'mileage',
                            'engine_volume',
                            'airbags'
            ]
            categorical_columns = ['manufacturer_country',
                            'category',
                            'leather_interior',
                            'fuel_type',
                            'turbo',
                            'gear_box_type',
                            'drive_wheels',
                            'steering_side',
                            'doors'
            ]

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(transformers=[
               ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                ],
                remainder='passthrough'
                )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and test data loaded")
           
            logging.info(
                f"Applying preprocessing object on training and test dataframe."
            )
            preprocessing_obj=self.get_data_transformer_object()
            
            train_arr=preprocessing_obj.fit_transform(train_df)
            test_arr=preprocessing_obj.transform(test_df)

            logging.info(f"Preprocessing object built.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            logging.info(f"Preprocessing object saved as pkl.")
        
        except Exception as e:
            raise CustomException(e,sys)