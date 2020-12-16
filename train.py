#!/usr/bin/env python

# Libraries
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import math
import json
import fasttext
import pickle
import warnings
warnings.filterwarnings('ignore')

# Data import
def data_import(filepath, col, col_new):

    df = pd.read_csv(filepath, encoding='latin1')
    df[col_new] = df[col]
    df[col_new] = df[col_new].astype(str)

    return df

# ext Cleaning
def preprocess(x, col, label):
    '''tokenize and normalize'''
    stop_words = set(stopwords.words('english')) 

    # convert to dataframe
    data = pd.DataFrame({'text': x[col], 'label': x[label]})

    # remove html
    data[col] = data.apply(lambda t: re.sub(r'https?://\S+|www\.\S+', '', str(t[col])), axis=1)

    # remove stopwords, number, and convert to lower case
    data[col] = data.apply(lambda r: ' '.join(w.lower() for w in r[col].split() if (w.lower() not in stop_words) & (w.isalpha())),axis=1)
    data[col] = data[data[col] != '']
    
    # discard NA reviews
    data = data.dropna()

    return data

# TF-IDF
def tfidf_transform(train, test, col):
    ngram = (1,2)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,ngram_range=ngram, stop_words='english')
    tfidf.fit_transform(train[col].values)

    # We transform each text into a vector
    vec_train = tfidf.transform(train[col].values)
    vec_test = tfidf.transform(test[col].values)

    # save best performing svm model
    with open('model/tfidf_vec.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    return vec_train, vec_test


def main():
    train = data_import("data/Corona_NLP_train.csv", 'OriginalTweet', 'text')
    test = data_import("data/Corona_NLP_test.csv", 'OriginalTweet','text')
    train_new = preprocess(train, 'text', 'Sentiment')
    test_new = preprocess(test, 'text', 'Sentiment')

    X_train, X_test = tfidf_transform(train_new, test_new, 'text')
    y_train = train_new.label.values
    y_test = test_new.label.values
    

    print('Train models!!!!')
    # Logistic

    lr1 = LogisticRegression(random_state=66,solver='lbfgs')  # fit logistic
    lr1.fit(X_train, y_train)
    y_pred = lr1.predict(X_test) # predict

    # evaluation metrics
    print('Logistic model 1:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    lr2 = LogisticRegression(random_state=66, C=15, penalty='l2',solver='lbfgs')  # fit logistic
    lr2.fit(X_train, y_train)
    y_pred = lr2.predict(X_test) # predict

    # evaluation metrics
    print('Logistic model 2:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    lr3 = LogisticRegression(random_state=66, C=10, penalty='l2',solver='lbfgs')
    lr3.fit(X_train, y_train)
    y_pred = lr3.predict(X_test) # predict

    # evaluation metrics
    print('Logistic model 3:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    lr4 = LogisticRegression(random_state=66, C=15, penalty='l2',solver='liblinear')
    lr4.fit(X_train, y_train)
    y_pred = lr4.predict(X_test) # predict

    # evaluation metrics
    print('Logistic model 4:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    lr5 = LogisticRegression(random_state=66, C=2, penalty='l1',solver='liblinear')
    lr5.fit(X_train, y_train)
    y_pred = lr5.predict(X_test) # predict

    # evaluation metrics
    print('Logistic model 5:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))


    # fasttext

    # fasttext requires data to be in the format of: __label__1 text
    train_fasttext = train_new.apply(lambda t: '__label__' + str(t['label']) + ' ' + str(t['text']), axis=1)
    test_fasttext = test_new.apply(lambda t: '__label__' + str(t['label']) + ' ' + str(t['text']), axis=1)
    train_fasttext.to_csv('fasttext_train.txt',index=False, header=False)
    test_fasttext.to_csv('fasttext_test.txt',index=False, header=False)

    # fasttext model - default
    ft_model1 = fasttext.train_supervised('fasttext_train.txt')

    # calculate evaluation metrics
    result = ft_model1.test('fasttext_test.txt')
    precision = result[1]
    recall = result[2]
    print('Fasttext model 1:')
    print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))

    # fasttext model - setting 1
    ft_model2 = fasttext.train_supervised('fasttext_train.txt',wordNgrams=2)
    result = ft_model2.test('fasttext_test.txt')
    precision = result[1]
    recall = result[2]
    print('Fasttext model 2:')
    print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))

    # fasttext model - setting 2
    ft_model3 = fasttext.train_supervised('fasttext_train.txt',lr=0.2, wordNgrams=2)
    result = ft_model3.test('fasttext_test.txt')
    precision = result[1]
    recall = result[2]
    print('Fasttext model 3:')
    print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))

    # fasttext model - setting 3
    ft_model4 = fasttext.train_supervised('fasttext_train.txt', lr=0.5, wordNgrams=2)
    result = ft_model4.test('fasttext_test.txt')
    precision = result[1]
    recall = result[2]
    print('Fasttext model 4:')
    print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))


    # SVM

    svm1 = LinearSVC(random_state=66)
    svm1.fit(X_train, y_train)
    y_pred = svm1.predict(X_test)
    
    print('SVM model 1:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    svm2 = LinearSVC(random_state=66, penalty='l2', C=10, loss='hinge')
    svm2.fit(X_train, y_train)
    y_pred = svm2.predict(X_test)
    
    print('SVM model 2:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    svm3 = LinearSVC(random_state=66, penalty='l2', loss='squared_hinge', dual=False)
    svm3.fit(X_train, y_train)
    y_pred = svm3.predict(X_test)
    
    print('SVM model 3:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    svm4 = LinearSVC(random_state=66, penalty='l1', loss='squared_hinge', dual=False)
    svm4.fit(X_train, y_train)
    y_pred = svm4.predict(X_test)
    
    print('SVM model 5:')
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))


    # Random Forest

    max_depth = [10,30,50]
    n_estimators = [200,500]
    grid_params ={'max_depth':max_depth,'n_estimators':n_estimators}

    RandomFoest_model = GridSearchCV(RandomForestClassifier(class_weight = 'balanced'), grid_params,
                      scoring = 'accuracy', cv=5,n_jobs=-1, return_train_score=True)
    RandomFoest_model.fit(X_train, y_train)

    results = pd.DataFrame.from_dict(RandomFoest_model.cv_results_)
    print(RandomFoest_model.best_estimator_)


    RandomFoest_model = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                          max_depth=50, n_estimators=200,  random_state=66, verbose=0)
    RandomFoest_model.fit(X_train,y_train)

    y_pred = RandomFoest_model.predict(X_test)
    print("Accuracy: %0.4f"%accuracy_score(y_test, y_pred))
    print("Micro-averaged F1 score: %0.4f"%f1_score(y_test, y_pred, average='micro'))

    #Store best model
    ft_model1.save_model('model/fasttext_model')


if __name__ == '__main__':
    main()
