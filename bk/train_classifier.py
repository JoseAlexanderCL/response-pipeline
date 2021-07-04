import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import libraries for training the model and getting evaluation metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import pickle



def load_data(database_filepath):
    """ Returns features (X), labels (y) and category names
    Take input to load the database from sqlite with message and categories.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df',engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns 
    return X, y, category_names



def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """ Pipeline with Tokenizer, TFID and MultiOutput Classiffier with Random Forest
    The hyperparameters were obtained with a Cross-Validation process

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20,max_features='auto',min_samples_leaf=4, random_state=1),n_jobs=-1))
    ])

    print(pipeline.get_params())
    
    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(0,36):
        print(f'category {i+1} {f1_score(Y_test.iloc[:,i],Y_pred[:,i]):.2f}  {accuracy_score(Y_test.iloc[:,i],Y_pred[:,i]):.2f} {precision_score(Y_test.iloc[:,i],Y_pred[:,i]):.2f}')




def save_model(model, model_filepath):
    """Writes RandomForest model into a .pkl
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))



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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()