#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request, url_for

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')



def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("checkpoints/model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/info')
def info():
    return flask.render_template('info.html')

@app.route('/mejor')
def mejor():
    return flask.render_template('mejor.html')

@app.route('/result',methods = ['POST'])

def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        try:
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list)
            if int(result)==0:
                prediction='Gana'
                valorkills=to_predict_list[0]
                valorasistencias=to_predict_list[1]
                valorhs=to_predict_list[2]
                if to_predict_list[3]==1:
                    valorvive="Si"
                else:
                    valorvive="No"
            elif int(result)==1:
                prediction='Pierde'
                valorkills=to_predict_list[0]
                valorasistencias=to_predict_list[1]
                valorhs=to_predict_list[2]
                if to_predict_list[3]==1:
                    valorvive="Si"
                else:
                    valorvive="No"
            else:
                prediction=f'{int(result)} No-definida'
                valorkills=to_predict_list[0]
                valorasistencias=to_predict_list[1]
                valorhs=to_predict_list[2]
                if to_predict_list[3]==1:
                    valorvive="Si"
                else:
                    valorvive="No"
        except ValueError:
            prediction='Error en el formato de los datos'
            valorkills=to_predict_list[0]
            valorasistencias=to_predict_list[1]
            valorhs=to_predict_list[2]
            if to_predict_list[3]==1:
                valorvive="Si"
            else:
                valorvive="No"

        return render_template("result.html", prediction=prediction, valorkills=valorkills, valorasistencias=valorasistencias, valorhs=valorhs, valorvive=valorvive)


if __name__=="__main__":

    app.run(port=5001)

url_for('static', filename='graf1.png')
url_for('static', filename='heatmap.png')