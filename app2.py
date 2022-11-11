# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:19:16 2022

@author: ragas
"""

import flask
import pandas as pd
from joblib import dump, load


with open(f'model.pkl', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('crop2.html'))

    if flask.request.method == 'POST':
        rooms = flask.request.form['N']
        bathroom = flask.request.form['K']
        landsize = flask.request.form['P']
        lattitude = flask.request.form['T']
        longtitude = flask.request.form['H']
        distance = flask.request.form['pH']
        car = flask.request.form['R']


        input_variables = pd.DataFrame([[rooms, bathroom, landsize, lattitude, longtitude, distance, car]],
                                       columns=['rooms', 'bathroom', 'landsize', 'lattitude', 'longtitude',
                                                'distance', 'car'],
                                       dtype='int',
                                       index=['input'])

        predictions = model.predict(input_variables)
        print(predictions)

        return flask.render_template('crop2.html', original_input={'N': rooms, 'K': bathroom, 'P': landsize, 'T': lattitude, 'H': longtitude, 'pH': distance, 'R': car},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)