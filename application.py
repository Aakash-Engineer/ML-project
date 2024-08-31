import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
from flask import render_template


application = Flask(__name__)
app = application


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'), 
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=int(request.form.get('writing_score')),
            reading_score=int(request.form.get('reading_score'))
        )
        
        pred_df = data.to_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(results[0], 2))
    
if __name__ == '__main__': 
    app.run()