
import pandas as pd
import numpy as np 

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import datetime
from sklearn.utils import shuffle
import scipy.sparse


class LogisticRegression:
    __weights = np.array([])
    __prediction = np.array([])
    
    def __init__(self):
        pass
    
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
    
    def predict_1(self, w, x):
        return self.sigmoid(np.dot(w,x))
    
    def find_error(self, target, prediction):    
        return target - prediction
    
    def get_numpy_data(self, feature_df, label_df):
        feature_df['_00'] = 1
        feature_matrix = feature_df.as_matrix()
        labels = label_df.as_matrix()
        return feature_matrix, labels
    
    def initialize(self, diamension):
        w = np.zeros(diamension)
        b = 0
        return w, b
    
    def gradient_ascent(self, feature, target, num_steps, learning_rate):
        weight, bias = self.initialize(feature.shape[1])

        for itr in range(num_steps):
            prediction = self.predict_1(feature, weight)
            error = self.find_error(target, prediction)
            gradient = np.dot(feature.T, error)
            weight_d = learning_rate * gradient
            weight += weight_d   
        self.__weights = weight
        return weight
    
    def fit(self, feature_df, label_df, num_steps, learning_rate):
        feature, target = self.get_numpy_data(feature_df, label_df)
        self.gradient_ascent(feature, target, num_steps, learning_rate)
    
    def predict(self, X_test):
        X_test['_00'] = 1
        feature_matrix = X_test.as_matrix()    
        self.__prediction = (self.predict_1(self.__weights, feature_matrix.T) > 0.5).astype(int)
        return self.__prediction
    
    def score(self, y_test):
        return (np.sum(self.__prediction == y_test) * 1.)/ y_test.shape[0]


data = pd.read_csv('farm-ads-data', header=None, delimiter='/n')


label = pd.read_csv('farm-ads-label', header=None, delimiter=' ')
label = label.drop([0], axis=1)


df = pd.concat([data, label], axis=1)

df = shuffle(df)

X_train_data, X_test, y_train_data, y_test = train_test_split(df[0],df[1],test_size=0.33, random_state=53, shuffle=False)
count = X_train_data.count()

scores = []

itr = 0
print datetime.datetime.now()
for frac in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
    itr += 1
    index = int(count * frac)
    X_train = X_train_data[:index]
    y_train = y_train_data[:index]
    
    count_vec = CountVectorizer(strip_accents=None)

    count_data = count_vec.fit_transform(X_train, y_train)

    scipy.sparse.save_npz('sparse_matrix_'+ str(frac) +'.npz', count_data)

    count_df = pd.DataFrame(count_data.A, columns=count_vec.get_feature_names())

    
    logistic = LogisticRegression()
    logistic.fit(count_df, y_train, 100, 0.001)
    
    logistic.predict(test_df)

    score = logistic.score(y_test)
    
    scores.append(score)
    print datetime.datetime.now(), itr, score
    
print scores

arr = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
df = pd.DataFrame(zip(arr, scores), columns=['a', 'b'])
%matplotlib inline
import matplotlib as plt
df.plot.scatter(x='a', y='b')
