from flask import Flask, render_template, request, Markup, flash, redirect, send_file, jsonify
from app import app
from app.DNN_s1 import train
import csv


@app.route('/')
def main():
	return render_template('index.html')

@app.route('/training')
def trainingForm():
	global acc, spec, sens
	acc, spec, sens = train(0.01, 'elu', 'elu', 'relu', 'relu', 0.3)
	return render_template('trainingfinish.html', acc=acc, spec=spec, sens=sens)

@app.route('/trainingfinish')
def trainingfinish():
    return render_template('trainingfinish.html')

@app.route("/api_dataset")
def api_dataset():
    data = {}
    csvs = [row for row in csv.reader(open('BreastSample_normal_vs_cancer.csv', 'r'))]
    data['data'] = csvs
    return jsonify(data)

@app.route('/dataset')
def dataset():
	return render_template('dataset.html')