import numpy as np
import pickle
from flask import Flask, request, render_template
import pandas as pd

# Load ML model
modelsc = pickle.load(open('best_model.sav', 'rb'))

# Create application
application = Flask(__name__)

app=application

@app.route('/')
def home(): 
    return render_template('index.html')


@app.route('/showsc')
def showsc():
    return render_template('Classifier.html')



@app.route('/predictsc', methods=['POST'])
def predictsc():

    
    features = [float(i) for i in request.form.values()]

    col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']
    output_data=pd.DataFrame([features],columns = col)
    #print(output_data)

    prediction = modelsc.predict(output_data)

    output = prediction
    #print(output)


    if output == 1:
        return render_template('Classifier.html',
                               result='Heart Disease Detected: ', positive='Yes', res2='Risk is HIGH')
    else:
        return render_template('Classifier.html',
                               result='Heart Disease Detected: ', positive='No', res2='Risk is LOW')


if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0')

