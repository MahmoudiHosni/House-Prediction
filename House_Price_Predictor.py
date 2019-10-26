from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json


# load the built-in model 
gbr = joblib.load('House_Price_Predictor_Model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def get_delay():
    result=request.form
    lotsize = result['lotsize']
    bedrooms = result['bedrooms']
    bathrms = result['bathrms']
    stories = result['stories']
    driveway = result['driveway']
    recroom = result['recroom']
    fullbase = result['fullbase']
    gashw = result['gashw']
    airco = result['airco']
    garagepl = result['garagepl']
    prefarea = result['prefarea']
    # we create a json object that will hold data from user inputs
    user_input = {'lotsize':5700, 'bedrooms':2, 'bathrms':2, 'stories':1, 'driveway':0,'recroom':1,'fullbase':0,
              'gashw':1,'airco':0,'garagepl':1,'prefarea':1}
    # encode the json object to one hot encoding so that it could fit our model
    a = input_to_one_hot(user_input)
    # get the price prediction
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)
    # return a json value
    return json.dumps({'price':price_pred});

if __name__ == '__main__':
    app.run(port=3333, debug=True)
