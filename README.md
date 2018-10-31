# Disaster-Response-Pipelines

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [About the data](#about_the_data)
4. [Files Description](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This project uses the following Python libraries you may need to install:

*pandas*

*sklearn*

*nltk*

*plotly*

*flask*

*sqlalchemy*

If you do not have any packages though the project, you can use `pip install package_name` to install them.In addition, NLTK data need to be downloaded separately, which is included in the program and may take a little time.

When completing package installaion, here is how to use this app:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## Project Motivation<a name="motivation"></a>

This project is from Udacity's data engineering project. It perform the pipeline of ETL and machine learning to classify information received about the disaster. It also includes a web app that displays a clean image of the data. At the same time, this web page is also used to classify the messages entered into the web page with the training model.

## About the data<a name="about_the_data"></a>

This data is about the classification of disaster messages. In the course of training, I found this data to be highly unbalanced. In particular, one part of the messages categories have many data, while the others may be few. The model performs well in the classification with more data, but poor for the messages categories with less data. 

I adopted over-sampling and under-sampling schemes, but they are not ideal. I think the main reason is that the dimensions of the data are too large. I used the weighted F1 score to evaluate the model, hoping that the model would pay more attention to the classification performance of all categories, rather than the overall accuracy.

## Files Description<a name="files"></a>

**run.py**: Web app script of the project, to start the processing in the background of the web app.

**go.html** and **master.html**: Front-end file of the web app.

**process_data.py**: ETL pipeline script, be used in cleaning the data for next ML step.

**train_classifier.py**: ML pipeline script, be used in training model for web app.

**ETL Pipeline Preparation.ipynb** and **ML Pipeline Preparation.ipynb**: Jupyter notebook file I use to explore the dataset, I already rewrite the code to python script so it may be useless for you.



## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the project. You can't use this for you Udacity supervise learning project. Otherwise, feel free to use the code here as you would like! 
