# Disaster Response Pipeline Project

### Summary:

This repo contains a worfklow for training a machine learning model on emergency related tweets, and doing inference on user input text. These results are deployed on a simple Flask app.

The first script in the model take two files with the tweets and their category (label for the model), then runs a ETL
process for cleaning the data. The second script takes this data and run a pipeline for transforming the text data and
then train a RandomForest model. This model is saved in a .pkl file.

Finally the run.py runs a Flask app with the design of the website and takes input data for estimating the category or label of the input.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgments

Most of the structure of the script is thanks to Udacity's recommendations. I really appreciate the provision of all the elements and resources that they make available for learning.

### Flask App Sample

![sample](sample.png)