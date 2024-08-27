import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from src.exception import CustomException

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_transformer(self):

        try:

            category_features = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 'test_preparation_course'
            ]
            numeric_features = ['reading_score', 'writing_score']

            numeric_transformer = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
                ('scaler', StandardScaler())
            ])

            category_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ])

            logging.info('Numerical and categorical transformers created')

            preprocessor = ColumnTransformer([
                ('num_pipeline', numeric_transformer, numeric_features),
                ('cat_pipeline', category_transformer, category_features)
            ], remainder='passthrough')

            return preprocessor
        
        except Exception as e:
            logging.info('Error in get_transformer method')
            raise CustomException(e, sys)
        
    def initiate_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test data loaded successfully')

            preprocessor_obj = self.get_transformer()
            logging.info('Preprocessor object created')

            x_train = train_df.drop('test_score', axis=1)
            y_train = train_df['test_score']

            x_test = test_df.drop('test_score', axis=1)
            y_test = test_df['test_score']

            x_train_transformed = preprocessor_obj.fit_transform(x_train)
            x_test_transformed = preprocessor_obj.transform(x_test)
            logging.info('Data transformed successfully')

            train_arr = np.concatenate((x_train_transformed, y_train.values.reshape(-1, 1)), axis=1)
            test_arr = np.concatenate((x_test_transformed, y_test.values.reshape(-1, 1)), axis=1)

            save_object(preprocessor_obj, self.transformation_config.preprocessor_obj_path)
            logging.info('preprocessor object saved successfully')

            return (train_arr, test_arr, self.transformation_config.preprocessor_obj_path)
        except Exception as e:
            logging.info('Error in initiate_transformation method')
            raise CustomException(e, sys)