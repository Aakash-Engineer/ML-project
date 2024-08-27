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
            params = {
                'LinearRegression': {
                    'fit_intercept': [True, False],
                },
                'RandomForest': {
                    'n_estimators': [20, 50, 100],
                    'max_depth': [5, 10, 15],
                    'bootstrap': [True, False]
                },
                'DecisionTree': {
                    'max_depth': [None, 5, 10, 15],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
                },
                'KNN': {
                    'n_neighbors': [5, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]
                },
                'SVR': {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto'],
                    'C': [0.1, 1, 10]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'max_depth': [3, 5, 7]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'max_depth': [3, 5, 7]
                },
                'CatBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'max_depth': [3, 5, 7]
                }
            }
            model_report: pd.DataFrame = evaluate_model(
                models = models,
                x_train = x_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                params = params
            ) 

            best_model = (model_report.sort_values(by='r2', ascending=False)).iloc[0]
            best_model_name = best_model['model']
            best_model_score = best_model['r2']

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)
            
            model = models[best_model_name]
            save_object(model, self.model_trainer_config.trained_model_file_path)


            logging.info('Model training completed')
            return model_report
   
        except Exception as e:
            logging.info('Error in initiate_model_training method')
            raise CustomException(e, sys)