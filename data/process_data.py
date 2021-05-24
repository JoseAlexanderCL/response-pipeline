import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load Data from CSV Files
    
    This function recieves the filepath of two CSV files, load the files
    and then return the merged dataframe.    
    
    The main attributes are Message and Categories (or labels of the message)
    
    """
        
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, sep = ",")
    df = messages.merge(categories)
    return df


def clean_data(df):
    """ Cleans Data for Pipeline
    
    Recieves a dataframe containing message and categories then apply
    cleaning steps:
        1. Expand the categories field
        2. Creates the column names for different categories
        3. Clean category values removing extra text
        4. Convert categories to number (0-1)
        5. Correct issues in "related"category
        6. Drops duplicated rows
    
    Returns cleaned dataframe df
    
    """
    
    
    categories = df['categories'].str.split(pat=";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:len(x)-2])
    categories.columns = category_colnames
    
    for i in categories:
        categories[i]=categories[i].astype(str).str[-1]
    
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    categories['related'] = categories['related'].map({0:0, 1:1, 2:0})
    
    df.drop(columns=['categories'],inplace=True)
    
    df = pd.concat([df,categories], axis=1,sort=False)
    
    
    #Removing duplicates
    
    df.drop_duplicates(inplace=True)
    
    return df




def save_data(df, database_filename):
    """ Save clean dataframe into sqlalchemy database
    
    Recieves database filename and uses sqlalchemy for
    saving a dataframe into "database_filename"
    
    """  
    
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('df', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()