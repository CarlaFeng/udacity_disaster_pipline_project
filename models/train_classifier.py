import sys
import sqlite3
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sqlalchemy import create_engine

    
def load_data(database_filepath):
   """
    功能：从数据库下载数据
    输入变量：数据库的路径
    输出变量：这里解释输出变量是什么
   """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    X = df.message.values
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = list(Y.columns)
    return X,Y,category_names

def tokenize(text):
   """
    功能：将一段文字按空格切开
    输入变量：一段字符串
    输出变量：单个文字的数组
   """
    text=re.sub(r"[^a-zA-Z0-9]", " ", text.lower())#normalize
    token=word_tokenize(text)#tokenize
    lemzer = WordNetLemmatizer()#lemmatizer
    clean=[]
    for tok in token:
        clean_tok = lemzer.lemmatize(tok).strip()
        clean.append(clean_tok)
    clean_words = [w for w in clean if w not in stopwords.words("english")]#remove stopword
    return clean_words


def build_model():
   """
    功能：建立模型管道
    输入变量：无
    输出变量：模型
   """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [1,2],
             'clf__estimator__min_samples_split':[2,3]}

    cv = GridSearchCV(pipeline, param_grid=parameters,return_train_score=True)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
   """
    功能：计算模型预测准确度，评估模型优劣
    输入变量：模型，测试数据集，分类
    输出变量：模型精确度
   """
    Y_pred = model.predict(X_test)
    df_Y_pred=pd.DataFrame(Y_pred,columns=category_names)
    for col in category_names:
        print('Report of '+col+':\n',classification_report(Y_test[col].values,df_Y_pred[col].values))
    


def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)


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
    #python models/train_classifier.py "data/disaster_data.db" "models/classifier.pkl"