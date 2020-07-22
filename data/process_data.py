import sys, os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    #功能：指定路径下读到csv数据
    #输入变量：csv数据集位置
    #输出变量：一个dataframe
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    # merge datasets
    df = messages.merge(categories, how='inner',on=['id'])
    return df


   
    
   
def clean_data(df):
    #功能：清洗数据
    #输入变量：dataframe
    #输出变量：去重，文字转数据后的数据集
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)
    category_colnames = list(map(lambda x: row[x][0][0:-2], [y for y in range(row.size)]))
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1,None,1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories = categories.replace(to_replace = 2, value =1) 
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates('message',inplace=True)
    return df


def save_data(df, database_filename):
    #功能：存储数据
    #输入变量：数据集，数据库名字
    #输出变量：无
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(database_filename, engine, if_exists='replace',index=False)
    

  
def main():
    path = os.path.split(os.path.realpath(__file__))[0]
    messages_filepath =  path+r"/disaster_messages.csv"
    categories_filepath = path+r"/disaster_categories.csv"
    database_filepath = path+r"/disaster_data.db"

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')



if __name__ == '__main__':
    main()
