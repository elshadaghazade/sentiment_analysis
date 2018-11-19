#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json as js
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


# In[5]:


json_data = None
data = pd.read_csv('./Sentiment-Analysis-Dataset/Sentiment Analysis Dataset.csv', error_bad_lines=False)


# In[6]:


stemmer = SnowballStemmer('english')
words = stopwords.words('english')


# In[10]:


data['cleaned'] = data['SentimentText'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub('[^a-zA-Z]', ' ', x).split() if i not in words]))


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.Sentiment, test_size=0.2)


# In[ ]:


pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
                    ('chi', SelectKBest(chi2, k=10000)),
                    ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']


# In[23]:


feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
featere_names = np.array(feature_names)

# target_names = ['1', '2', '3', '4', '5']
# print("top 10 keywords per class:")
# for i, label in enumerate(target_names):
#     top10 = np.argsort(clf.coef_[i])[-10:]
#     print(top10)
# #     print("%s %s" % (label, " ".join(feature_names[top10])))
    


# In[28]:


print("accuracy score:", model.score(X_test, y_test))
print(model.predict(['that was an awesome place. Good food!', "im dying", "incredible movie", "awful"]))

