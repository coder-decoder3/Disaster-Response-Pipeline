# import required dependencies
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    I/P:
        messages_filepath =  File path of the messages dataset.
        categories_filepath = File path for category records.
    O/P:
        dataframe = Merged DataSet
    '''
        
    # Loading messages data from dataset    
    messages = pd.read_csv(messages_filepath)
    
    # Loading categories data from dataset
    categories = pd.read_csv(categories_filepath)
    # Let's merge the DataSets
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    return df


def clean_data(dataframe):
    '''
    input:
        dataframe = merged dataset.
    output:
        dataframe = Cleaning dataset.
    '''
    # Create a dataframe with 36 categorical columns 
    categories = dataframe.categories.str.split(';', expand = True)
    # Let's select the first row in the categories
    row = categories.loc[0]
    # Use this line for extract the list of new column names for categories. 
    # One way is to use a lambda function that takes all
    # to the second-to-last character of each sliced string
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    # rename/change the 'categories' of columns
    categories.columns = category_colnames
    # Let's convert the category values to just numbers 1 or 0
    for column in categories:
    # set every cost to be the closing man or woman of the string
        categories[column] = categories[column].astype(str).str[-1]
        # let's convert/change column vals from string to nums
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from 'dataframe'
    dataframe.drop('categories', axis = 1, inplace = True)
    # Concatenate the original dataframe to the new "categories" dataframe.
    dataframe = pd.concat([dataframe, categories], axis = 1)
    # let's drop the duplicates
    dataframe.drop_duplicates(subset = 'id', inplace = True)
    return dataframe


def save_data(dframe, database_filepath):
    # Let's create the engine
    engine = create_engine('sqlite:///' + database_filepath)
    # Let's now save the clean dataset into an Database (SQLite)
    dframe.to_sql('CleanTable', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Enter news and category file paths '\
                 'Record the following as the first or second argument: '\
                 'and  the file path of the database to store the cleaned data '\
                 'as the third argument. \n\nExample: python process_data.py Disaster_Messages.csv Disaster_Categories.csv '\
                'DisasterResponse.db')


if __name__ == '__main__':
    main()