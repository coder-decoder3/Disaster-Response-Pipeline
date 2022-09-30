# import required dependencies
import re
import ssl
import sys
import nltk
import joblib
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import recall_score, classification_report, precision_score, f1_score, accuracy_score


nltk.set_proxy('http://proxy.example.com:3128', ('USERNAME', 'PASSWORD'))
nltk.download()
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    '''
    I/N:
        database_filepath =  SQL database file path
    O/P:
    X = Features DFrame
    Y =  Target DFrame
    category_names list = Target Labels 
    '''
    # Loading from DB
    sql_engine = create_engine('sqlite:///'+ database_filepath)
    dframe = pd.read_sql_table('CleanTable', sql_engine)
    X = dframe.message.values
    Y = dframe[dframe.columns[4:]].values
    category_names = list(dframe.columns[4:])
    return X, Y, category_names


def tokenize(input_text):
    '''
    I/N:
        text = tokenized message.
    O/P:
        clean_tokens = List after tokenization.
    '''
    input_text = re.sub(r"[^a-zA-Z0-9]", " ", input_text.lower())
    tokens = word_tokenize(input_text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    create_pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    parameters = {'clf__estimator__estimator__C': [1, 2, 5]}
    cv_grid =  GridSearchCV(create_pipeline, param_grid=parameters, verbose=3, cv = 5)
    return cv_grid


def evaluate_model(model, X_test, y_test, category_names):
    # Predict Msg Categories
    y_pred = model.predict(X_test)
    # Print accuracy score, accuracy score, recall score and f1_score for each category
    for each in range(0, len(category_names)):
        print(category_names[i])
        print("\tAccuracy: {:.4f}\t|| Precision: {:.4f}\t|| Recall: {:.4f}\t|| F1_score: {:.4f}".format(
            accuracy_score(y_test[:, each], y_pred[:, each]),
            precision_score(y_test[:, each], y_pred[:, each], average='weighted'),
            recall_score(y_test[:, each], y_pred[:, each], average='weighted'),
            f1_score(y_test[:, each], y_pred[:, each], average='weighted')
        ))

def save_model(model, model_file_path):
    with open(model_file_path, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please specify the file_path of the disaster report database '\
              'as the first parameter and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()