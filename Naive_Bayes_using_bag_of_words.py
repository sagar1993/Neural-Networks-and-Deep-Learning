import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
class NaiveBayes:
    __data_sum_classes = []
    __priors = {}
    __count_labels = None
    __all_result = []
    def __init__(self):
        pass
    
    def fit(self, train_x, train_y):
        liklihood = {}
        df = pd.concat((train_x, train_y), axis =1)
        total = df.shape[0]
        columns = df.columns
        data_sum = df.iloc[:,:-1].sum()
        
        self.__count_labels = df.iloc[:,-1].unique()
        
        
        for i in self.__count_labels:
            data = df[df.iloc[:,-1] == i]
            count = data.shape[0]
            prior = (count * 1.)/ total
            self.__priors[i] = prior
            numerator = (df[df.iloc[:,-1] == i].iloc[:,:-1].sum() + 1.0)
            denominator = (data_sum + len(self.__count_labels))
            self.__data_sum_classes.append(numerator/ denominator)
        
    def predict(self, test_y):
        all_result = []
        total = test_y.shape[0]
        for i in range(total):
            row = test_y[i:i+1]
            row_sum = row.sum()
            row_sum = (row_sum >= 1).astype(int)
            _max = None
            index = 0
            for i in self.__count_labels:
                product = row_sum * self.__data_sum_classes[i]
                product_1 = product[product > 0.0]
                val =  np.prod(product_1) * self.__priors[i]
                if not _max:
                    _max = val
                    index = i
                if _max > val:    
                    _max = val
                    index = i
            all_result.append(index)
        self.__add_result = np.array(all_result)
        return self.__add_result
    
    def score(self, test_y):
        return (np.sum(self.__add_result == test_y) * 1.)/ test_y.shape[0]
    
    
data = pd.read_csv('farm-ads-data', header=None, delimiter='/n')


label = pd.read_csv('farm-ads-label', header=None, delimiter=' ')
label = label.drop([0], axis=1)


df = pd.concat([data, label], axis=1)


X_train_data, X_test, y_train_data, y_test = train_test_split(df[0],df[1],test_size=0.33, random_state=53, shuffle=False)
count = X_train_data.count()

scores = []

itr = 0
# [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
for frac in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
    itr += 1
    index = int(count * frac)
    #print index
    X_train = X_train_data[:index]
    y_train = y_train_data[:index]
    
    count_vec = CountVectorizer(strip_accents=None)

    count_data = count_vec.fit_transform(X_train, y_train)

    count_df = pd.DataFrame(count_data.A, columns=count_vec.get_feature_names())

    
    naive = NaiveBayes()
    naive.fit(count_df, y_train)

    data = count_vec.transform(X_test)
    test_df = pd.DataFrame(data.A, columns=count_vec.get_feature_names())
    
    #scores.append(clf.score(test_df, y_test))
    
    naive.predict(test_df)

    score = naive.score(y_test)

    scores.append(score)
    print datetime.datetime.now(), itr, score
    
print scores

arr = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
df = pd.DataFrame(zip(arr, scores), columns=['a', 'b'])
%matplotlib inline
import matplotlib as plt
df.plot.scatter(x='a', y='b')
