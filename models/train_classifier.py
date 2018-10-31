import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk import pos_tag
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

def load_data(database_filepath):
    '''
    This function read data from database and split data to X and Y.
    
    INPUT:
    database_filepath - the database file path.
    
    OUTPUT:
    X - a dataframe include messages. 
    Y - a dataframe include messages' categories. 
    cat - a list include messages category types. 
    '''
    # Create engine to connect the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_cat', engine)
    # Split data to X, Y, category types
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


def tokenize(text):
    '''
    This function clean the message and tokenize it to words list.
    
    INPUT:
    text - the text need to be tokenized.
    
    OUTPUT:
    clean_tokens - a list of words included in the text.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    This class is a transformer to be used to decide whether the starting word is verb.
    '''
    def starting_verb(self, text):
        '''
        This function spot whether the text has a starting verb.

        INPUT:
        text - a text we need to get the starting word.

        OUTPUT:
        has_verb: a flag whether the text has a starting verb.
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        This function detect whether each of X has a starting verb.

        INPUT:
        X - a nparray text we need to get the starting word.

        OUTPUT:
        has_verb: a flag nparray whether the text has a starting verb.
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    This function build the mechine learning pipeline and gridsearch model.

    INPUT:
    None.

    OUTPUT:
    cv_model: a text process and mechine learning pipeline gridsearch model.
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 200],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1}
        )
    }

    cv_model = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1, verbose=1, scoring='f1_weighted')
    return cv_model

def tune_model(cv_model, X_train, Y_train):
    '''
    This function train every model in cv_model to tune the model and return the best model.

    INPUT:
    cv_model - a GridSearchCV model need to be trained and tuned.
    X_train - a dataframe of predictor variable for train.
    Y_train - a dataframe of response variables for train.

    OUTPUT:
    model: a model which is the best one in fine tune.
    '''
    cv_fit = cv_model.fit(X_train, Y_train)
    return cv_fit.best_estimator_


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function print the model performence by .

    INPUT:
    cv_model - a GridSearchCV model need to be trained and tuned.
    X_train - a dataframe of predictor variable for train.
    Y_train - a dataframe of response variables for train.

    OUTPUT:
    None.
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    This function save the model to a file.

    INPUT:
    model - a model need to be saved.
    model_filepath - a file path where the model will save to.

    OUTPUT:
    None.
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv_model = build_model()

        print('Tunning model...')
        model = tune_model(cv_model, X_train, Y_train)
        
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