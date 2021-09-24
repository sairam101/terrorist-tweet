# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
pd.set_option('display.max_columns',None)
import nltk
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle 


stop = stopwords.words('english')
terroriest = pd.read_csv('twitterdataset.csv',header=None) 
normal = pd.read_csv('normal.csv',encoding='ISO-8859-1',header=None)
terroriest.columns = ['target','date','text'] 
terroriest = terroriest[['target','text']]
terroriest['target'] = 1
pd.DataFrame(terroriest['text'])
normal = normal.iloc[:3000,4:]
normal.columns = ['target','text']
normal['target'] = 0 
#print(normal)
df = pd.concat([normal,terroriest],axis=0)
df['target'].value_counts()
df['text']
# Stopwords
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

def clean(text):
    a = []
    for t in text.split():
        if t[0:5] != "https" and t[0:4] != "http":
            a.append(t.lower())
        elif t[0] not in ['@','#'] :
            a.append(t.lower())
    return ' '.join(a)
    
        
def punctuation_removal(df):
    df['text'] = df['text'].str.replace('[^\w\s]','')    
    return df


df = punctuation_removal(df)   
# Stopwords
df['text'] = df['text'].apply(lambda x: clean(x))
acc = []
xtrain, xtest = train_test_split(df,test_size=0.2)
CV = CountVectorizer(max_features=5048,ngram_range=(1,3))
train = CV.fit_transform(xtrain['text']).toarray()
test = CV.transform(xtest['text']).toarray()
    
# Model Training 
# Multinomial NaiveBayes
def naiveBayes():
    global acc
    model_n = MultinomialNB(alpha=1)
    model_n.fit(train,xtrain['target'])
    #Model Testing
    ypredict = model_n.predict(test)
    accuracy_score(xtest['target'],ypredict)
    acc.append(accuracy_score(xtest['target'],ypredict))
    my = 'pakistan fraud crime '
    new = pd.Series(my)
    demo = CV.transform(new).toarray()
    #print(demo)
    model_n.predict_proba(demo)[:,0]
    terrorist_model = pickle.dump(model_n, open("bayes_model.npy", "wb"))
    


#Model 2
# LogisticRegression
from sklearn.linear_model import LogisticRegression
def logisticRegression():
    global acc
    model_l = LogisticRegression()
    model_l.fit(train,xtrain['target'])
    #Model Testing
    ypredict = model_l.predict(test)
    accuracy_score(xtest['target'],ypredict)
    acc.append(accuracy_score(xtest['target'],ypredict))
    my = 'pakistan fraud crime'
    new = pd.Series(my)
    demo = CV.transform(new).toarray()
    terrorist_model = pickle.dump(model_l, open("logistic_model.npy", "wb"))



#model 3
#SVM 

from sklearn.svm import SVC
def svm():
    global acc
    model_s = SVC()
    model_s.fit(train,xtrain['target'])
    #Model Testing
    ypredict = model_s.predict(test)
    accuracy_score(xtest['target'],ypredict)
    acc.append(accuracy_score(xtest['target'],ypredict))
    my = 'pakistan fraud crime '
    new = pd.Series(my)
    demo = CV.transform(new).toarray()
    terrorist_model = pickle.dump(model_s, open("terriorist_model.npy", "wb"))

    
svm()
logisticRegression()
naiveBayes()
print("accuracy",acc)
