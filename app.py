import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
#initialize the flask app
app=Flask(__name__)

#loading the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    model=pickle.load(file)


#Home Page 
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #taking inputs from user
    N=float(request.form['N'])
    P=float(request.form['P'])
    K=float(request.form['K'])
    temperature=float(request.form['temperature'])
    humidity=float(request.form['humidity'])
    ph=float(request.form['ph'])
    rainfall=float(request.form['rainfall'])

    # numpy array for the model
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict the crop using the model
    recommended_crop = model.predict(input_data)[0]
    
    # Return the result to the webpage
    return render_template('index.html', prediction_text=f'It is recommended to grow {recommended_crop} on your farm')


#run the app

if __name__=="__main__":
    app.run(debug=True)