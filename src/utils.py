import os
import sys
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def save_object(obj, path):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.error('Error in save_object method')
        raise CustomException(e, sys)

def evaluate_model(models, x_train, y_train, x_test, y_test, params):
    try:
        model_names = []
        mse_scores = []
        mae_scores = []
        r2_scores = []
        cross_scores = []

        for name, model in models.items():

            grid_search = GridSearchCV(model, params[name], cv=5, scoring='r2', refit=True)
            grid_search.fit(x_train, y_train)

            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cross_val = cross_val_score(model, x_test, y_test, cv=10, scoring='r2')
            
            models[name] = best_model
            model_names.append(name)
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            cross_scores.append(cross_val)
        
        report = pd.DataFrame({
            'model':model_names,
            'r2': r2_scores,
            'cross_val_r2': cross_scores,
            'mse': mae_scores,
            'mae': mae_scores
        })

        return report
    
    except Exception as e:
        logging.info('Error in evaluate model function')
        raise CustomException(e, sys)
    
def load_object(path):
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        logging.error('Error in load_object method')
        raise CustomException(e, sys)