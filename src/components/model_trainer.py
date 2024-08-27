import os
import sys
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor, 
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Model training initiated')
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'KNN': KNeighborsRegressor(),
                'SVR': SVR(),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }
            model_report: pd.DataFrame = evaluate_model(
                models = models,
                x_train = x_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test
            ) 

            best_model = (model_report.sort_values(by='r2', ascending=False)).iloc[0]
            best_model_name = best_model['model']
            best_model_score = best_model['r2']

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)
            
            model = models[best_model_name]
            save_object(model, self.model_trainer_config.trained_model_file_path)
   
        except Exception as e:
            logging.info('Error in initiate_model_training method')
            raise CustomException(e, sys)