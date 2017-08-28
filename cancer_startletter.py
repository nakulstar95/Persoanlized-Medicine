#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:21:24 2017

@author: nakulsaiadapala
"""


import pandas as pd
import numpy as np
import os
os.chdir("/Users/nakulsaiadapala/Downloads/cancer")
train_var = pd.read_csv("training_variants.csv")
train_text = train_text_df = pd.read_csv("training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_var = pd.read_csv('test_variants')
test_text = pd.read_csv('test_text',sep = '\|\|', engine = "python",skiprows = 1,names=["ID", "Text"])

train_var.head()
test_var.head()
train_text.head()

train_df = pd.merge(left = train_var , right = train_text , on = 'ID' , how = 'left')
train_df.head()
test_df = pd.merge(left = test_var, right = test_text, on = 'ID', how = 'left')
test_df.head()

train_features = train_df.drop('Class',axis = 1)
all_data = pd.concat([train_features,test_df])
all_data['start_letter'] = all_data['Variation'].astype(str).str[0]
all_data['last_letter'] = all_data['Variation'].astype(str).str[-1]
all_data['combined'] = all_data['start_letter'].map(str)+all_data['last_letter']
all_data['G_start'] = all_data['Gene'].astype(str).str[0]
all_data['G_last'] = all_data['Gene'].astype(str).str[-1]
all_data['G'] = all_data['G_start'].map(str)+all_data['G_last']

all_data = all_data.drop(['G_start','G_last','start_letter','last_letter','Gene','Variation'],axis = 1)
all_data.head()

from sklearn.feature_extraction.text import TfidfVectorizer
sentences = all_data['Text']
vect = TfidfVectorizer(stop_words = 'english')
sentence_vectors = vect.fit_transform(sentences)


combined_vectors = vect.fit_transform(all_data['combined'])
Gene_vectors = vect.fit_transform(all_data['G'])


from scipy.sparse import hstack
total_vectors = hstack([sentence_vectors,combined_vectors,Gene_vectors])

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(200)

total_vectors = svd.fit_transform(total_vectors)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
def neural_model():
    model = Sequential()
    model.add(Dense(512, input_dim=200, init='normal', activation='relu'))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dense(9, init='normal', activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_y = train_df['Class'].values
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)
dummy_y = np_utils.to_categorical(encoded_y)
estimator = KerasClassifier(build_fn=neural_model, epochs=10, batch_size=64)
estimator.fit(total_vectors[0:3321], dummy_y, validation_split=0.05)
y_pred = estimator.predict_proba(total_vectors[3321:])
submission = pd.DataFrame(y_pred)
test_index = test_var['ID'].values
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv("nakul_submission.csv",index=False)