import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig



@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info('Initiating data ingestion')
        try:
            df = pd.read_csv('notebooks/data/stud.csv')
            logging.info('Data ingestion successful')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            logging.error('Data ingestion failed')
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    report = model_trainer.initiate_model_training(train_arr, test_arr)

