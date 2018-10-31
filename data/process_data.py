import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def load_data(messages_filepath, categories_filepath):
    '''
    This function read messages file and categories file. Then merge them to a dataframe.
    
    INPUT:
    messages_filepath - the path where the messages file is.
    categories_filepath - the path where the categories file is.
    
    OUTPUT:
    df - a dataframe include messages and categories information. 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    This function clean the data. All categories text is divided into separate columns.
    Each column value representing whether or not it belongs to this category.
    
    INPUT:
    df - a dataframe need to be cleaned.
    
    OUTPUT:
    new_df - a dataframe have already been cleaned.
    '''
    # Split the categories into columns by ';'
    categories = df.categories.str.split(';', expand=True)

    # Use one row to extract a list of new column names for categories
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].str.get(-1)
        categories[column] = categories[column].astype(int)
    # Some data in 'related' is 2, replace it into 1
    categories.loc[categories.related == 2, 'related'] = 1

    # drop the original categories column from `df` and concatenate new categories columns
    new_df = df.drop('categories', axis=1)
    new_df = pd.concat([new_df, categories], axis=1)

    # Remove dupilicates
    new_df.drop_duplicates(inplace=True)

    return  new_df


def save_data(df, database_filename):
    '''
    This function save the data to database, table name is messages_cat.
    It'll drop the table if exist.
    
    INPUT:
    df - a dataframe need to be saved.
    database_filename - a database file name which the data will be saved into.
    
    OUTPUT:
    None.
    '''
    # Create engine to database
    engine = create_engine('sqlite:///' + database_filename)
    DB_Session = sessionmaker(bind=engine)
    session = DB_Session()
    # Drop the table if exist
    session.execute('DROP TABLE IF EXISTS messages_cat')
    # Save the data
    df.to_sql('messages_cat', engine, index=False)  


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