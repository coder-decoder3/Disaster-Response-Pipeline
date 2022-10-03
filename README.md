#  Udacity Project for Disaster Relief Pipeline

## Installation
 Any Python version 3.5+
 
 ML libraries: NumPy, SciPy, Pandas, Sciki-Learn
 
 Natural Language Processing Library: NLTK
 
 Web apps and data visualization: Flask, Plotly
 
 Library for loading and saving models: Pickle
 
 SQLlite database library: SQLalchemy

## Motivation
In this Udacity DRP Project, we applied data engineering, natural language processing, and machine learning to analyze message data sent by people  during a disaster and created a model for an API to classify disaster messages. These messages may be sent to appropriate disaster relief organizations.

## Structure of Files

 - Application

| | - template

| | |- master.html # web application main page

| | |- go.html # Web application classification result page

|- run.py # Flask file to run app


 - Data

|- Disaster_categories.csv # Dataset

|- Disaster_Messages.csv # Dataset

|- process_data.py

|- CleanDatabase.db # Database to store clean data


 - Model

|- train_classifier.py

|- cv_AdaBoostr.pkl # saved model 


 - README.md


 ### Instructions:
1. Run the following commands from the  root of your project to set up your database and model:
- To run an ETL pipeline that cleans the data and saves it to the database
cd data
Run the following command using python:
'python disaster_messages.csv
        process_data.py 
        DisasterResponse.db'
        disaster_categories.csv 

 - To run an ML pipeline that trains  and saves a classifier
'python models/train_classifier.py models/cv_AdaBoost.pkl data/DisasterResponse.db'

 2. Run the flask web app by running the following command in your app's directory: 'python run.py'

 3. Browse to http://0.0.0.0:3001/ 
