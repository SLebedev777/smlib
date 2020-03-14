#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:32:26 2019

@author: sl
"""
import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB as skBNB
from smlib.bayes.nb import BernoulliNB

X_train = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 1]
        ])

y_train = np.array([
        1,
        0,
        1,
        0,
        0])

X_test = np.array([
        [1, 1, 0],
        [0, 1, 0],
        ])

y_test = np.array([
        1,
        0])


for classifier in [skBNB(), BernoulliNB()]:
    classifier.fit(X_train, y_train)
   
    print(classifier)
    expected = y_test
    predicted = classifier.predict(X_test)
    print(classification_report(expected, predicted))

print('-'*50)

X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
skbnb = skBNB()
skbnb.fit(X, Y)
print(skbnb.predict(X[2:3]))

bnb = BernoulliNB()
bnb.fit(X, Y)
print(bnb.predict(X[2:3]))

print('-'*50)
print('classifying 20newsgroups')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

remove = ('headers', 'footers', 'quotes')
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

y_train, y_test = data_train.target, data_test.target

vectorizer = TfidfVectorizer(max_df=0.5, use_idf=False, norm=None,
                                 stop_words='english', binary=True)
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()

for classifier in [skBNB(), BernoulliNB()]:
    print(classifier)
    print('fit...')
    classifier.fit(X_train, y_train)
   
    expected = y_test
    print('predict...')
    predicted = classifier.predict(X_test)
    print(classification_report(expected, predicted))

