from __future__ import division

from flask import render_template, request, Response, jsonify,redirect,url_for,flash
from werkzeug.utils import secure_filename

from app import app

import json
import psycopg2
import psycopg2.extras
import os
import pandas as pd
import hashlib
import datetime
from datetime import date
import numpy as np
from subprocess import Popen
import shlex
import sys
import requests
import threading
from json2table import convert
from flask import send_from_directory

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from learn import forall as fa
from learn import utils

TRAINING_DATA={}
TESTING_DATA={}


ALLOWED_EXTENSIONS=set(['txt','csv'])
SECRET_KEY='ml4all'
app.secret_key='ml4all'

p="global variable for the vis server"

@app.route('/index')
def index():
	return render_template('home.html')

@app.route('/')
def dataset():
   return render_template('home.html')

@app.route('/method')
def method():
	return render_template('method.html')


def to_csv(d, fields):
	d.insert(0, fields)
	return Response('\n'.join([",".join(map(str, e)) for e in d]), mimetype='text/csv')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/dataset',methods=['POST'])
def upload_file():
	global p
	train_file_name = 'train'
	test_file_name ='test'
	error=None
	if request.method == 'POST':
		try:
			p.terminate()
			print("Shiny server killed.")
		except Exception as e:
			print(e)
			print("Did not find a Shiny server to kill...")

		# check if the post request has the file part
		if train_file_name not in request.files or test_file_name not in request.files:
			#flash('No file part')
			error='Kindly upload both training and testing files'
			#print("helllllo")
			#print(request.url)
			flash("load files")
			#return redirect(request.url)
			return render_template('home.html',error=error)


		file = request.files[train_file_name]

		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':

			print(request.url)
			error='Kindly upload both training and testing files'

			flash('No selected files')
			return redirect(request.url)
			#return render_template('home.html',error=error)

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			print(filename)
			print(os.path.abspath(os.path.join('app/','uploads/')))
			#file.save(os.path.abspath(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename)))
			file.save(os.path.abspath(os.path.join('app/','uploads/', filename)))
			print("done")
			## convert file to pandas dataframe
			#df_train=pd.read_csv(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename))
			df_train=pd.read_csv(os.path.join('app/','uploads/', filename))

			print("df_train1",df_train.head(5))

			## hash the pd , change to binary --> get fom Jason
			temp_hash=pd.util.hash_pandas_object(df_train)
			hash_train = hashlib.sha256(str(temp_hash).encode('utf-8','ignore')).hexdigest()

			#Save train data in /uploads folder
			train_file_name=filename
			os.system("mv app/uploads/" + filename + " " + "app/uploads/" + hash_train + ".csv")
			## update dict ---> key:hash ,value: dataframe
			#TRAINING_DATA[hash_train]=df_train

		## For the test file
		file = request.files[test_file_name]

			# if user does not select file, browser also
			# submit a empty part without filename
		if file.filename == '':
			print(request_url)
			flash('No selected files')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#file.save(os.path.abspath(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename)))
			file.save(os.path.abspath(os.path.join('app/','uploads/', filename)))

			## convert file to pandas dataframe
			#df_test=pd.read_csv(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename))
			df_test=pd.read_csv(os.path.join('app/','uploads/', filename))
			print("df_test1",df_test.head(5))

			## hash the pd , change to binary --> get fom Jason
			temp_hash=pd.util.hash_pandas_object(df_test)
			hash_test = hashlib.sha256(str(temp_hash).encode('utf-8','ignore')).hexdigest()

			# Save test data in /uploads folder
			test_file_name=filename
			os.system("mv app/uploads/" + filename + " " + "app/uploads/" + hash_test + ".csv")

		# Pass datasets to Shiny app
		p = Popen(shlex.split("Rscript app/shiny/shiny.R " + hash_train + ".csv " + hash_test + ".csv " + train_file_name + " " + test_file_name))
		return(jsonify({"hash_train": hash_train, "hash_test": hash_test}))


@app.route('/shiny',methods=['GET'])
def check_shiny():
	# Check if Shiny server is up
	response_code = 0
	import time
	while response_code != 200:
		try:
			r = requests.head("http://127.0.0.1:2326")
			response_code = r.status_code
			print(r.status_code)
		except requests.ConnectionError:
			time.sleep(0.1)
			print("Trying to connect to Shiny server.")
			pass

	return("Shiny server is up.")


@app.route('/predict/<hash_train>_<hash_test>', methods=['GET'])
def run_prediction(hash_train, hash_test):
	train_file=hash_train + ".csv"
	test_file=hash_test + ".csv"

	# Make predictions
	df_train=pd.read_csv(os.path.join('app/','uploads/', train_file))
	df_test=pd.read_csv(os.path.join('app/','uploads/', test_file))
	X, y = utils.X_y_split(X_train=df_train, X_test=df_test)
	model = fa.All()
	model.fit(X, y)

	# Append prediction column to test set at column 1
	predictions = model.predict(df_test)
	prediction_name="predict_" + model.target_name
	df_test.insert(0, prediction_name, predictions)

	# Get current time
	from datetime import datetime
	t_current=datetime.now().strftime('%Y%m%d%H%M%S')

	# Save output file in /downloads folder
	df_test.to_csv("app/downloads/" + "Predict" + "_" + model.target_name + "_" + t_current + ".csv", index=False)

	# Add model.display_score to JSON and round values
	output_dic = {"Overall score" : model.display_score,
				#   model.understandable_metric_name : round(model.understandable_metric_value, 2),
				  " " : model.understandable_metric_description}

	# Build HTML table
	build_direction = "LEFT_TO_RIGHT"
	table_attributes = {"style" : "width:60%"}
	display_score = convert(output_dic, build_direction=build_direction, table_attributes=table_attributes)

	return(jsonify({"fileid": "Predict" + "_" + model.target_name + "_" + t_current, "performance": display_score}))


@app.route('/download/<hashid>',methods=['GET'])
def prediction_test(hashid):
	return send_from_directory(directory=os.path.abspath(os.path.join('app/downloads/')), filename=hashid + ".csv")
