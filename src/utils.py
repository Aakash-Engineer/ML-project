import os
import sys
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def save_object(obj, path):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.error('Error in save_object method')
        raise CustomException(e, sys)
    