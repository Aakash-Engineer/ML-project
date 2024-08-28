import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
            
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data = preprocessor.transform(features)
            prediction = model.predict(data)
            return prediction   
        except Exception as e:
            logging.info('Error in predict method')
            raise CustomException(e, sys)
class CustomData:

    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            writing_score: int,
            reading_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score
    
    def to_frame(self):
        try:
            frame = pd.DataFrame({
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'writing_score': [self.writing_score],
                'reading_score': [self.reading_score]
            })
            return frame
        except Exception as e:
            logging.info('Error in converting to frame in CustomData  class')
            raise CustomException(e, sys)