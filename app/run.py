import re
import ssl
import nltk
import json
import joblib
import plotly
import numpy as np
import pandas as pd
from flask import Flask
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from flask import render_template, request, jsonify
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load DB Data
db_engine = create_engine('sqlite:///../data/CleanDatabase.db')
df = pd.read_sql_table('CleanTable', db_engine)

# Load Models Data
model = joblib.load("../models/cv_AdaBoost.pkl")


# Display Index Page which shows cool visuals & takes user's input text & show's output from model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Here's an example - modify it to extract data for your own visuals 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Show distribution of different category
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))
    
    # extract data exclude related
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories_mean.index)
      
    # create visuals
    graphs_data = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'The Distribution of Messages Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
         {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
        
    
    # JSon Encoder Graphs Data
    ids = ["graph-{}".format(each) for each, _ in enumerate(graphs_data)]
    graphJSON = json.dumps(graphs_data, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render Template(WebPage (Graph))
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# A web page that handles user requests and displays model results.
@app.route('/go')
def go():
    # store user input in query 
    query = request.args.get('query', '') 

    # Use the model to predict the query classification 
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This renders go.html. Please see this file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
