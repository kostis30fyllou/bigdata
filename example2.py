import gensim
from gensim import corpora
from pprint import pprint
from utils import *
from sklearn import svm
import numpy as np
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
 
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from gensim.models import Word2Vec


documents = read_csv("train_set.csv").head(1000)
documents2 = read_csv("train_set.csv").head(1000)
test = read_csv("test_set.csv").head(1000).Content

#test2 = read_csv("test_set.csv").iloc[1].Content

##print("---------------------SVM-TfidfVectorizer-------------")
##
##vectorizer = TfidfVectorizer()
##vectorizer.fit(documents)
##
##X = vectorizer.transform(documents)
##
##
##Y=vectorizer.transform(test)
##
##clf = svm.SVC(gamma = 'scale')
##clf.fit(X,documents2)
##
##print(clf.predict(Y))

##print("--------------------SVM-CountVectorizer-------------")
##vectorizer = CountVectorizer()
##X = vectorizer.fit_transform(documents)
##
##Y = vectorizer.transform(test)
##clf = svm.SVC(gamma = 'scale')
##clf.fit(X,documents2)
##
##print(clf.predict(Y))
##
##print("---------------------RandomfOREST-CountVectorizer-------------")
##
##vectorizer = CountVectorizer()
##X = vectorizer.fit_transform(documents)
##
##Y = vectorizer.transform(test)
##clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
##
##clf.fit(X,documents2)
##
##print(clf.predict(Y))
##
##print("---------------------SVD-RandomfOREST -------------")

##vectorizer = CountVectorizer()
##X = vectorizer.fit_transform(documents)
##
##Y = vectorizer.transform(test)
##
##svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
##
##X=svd.fit_transform(X)
##Y = svd.transform(Y)
##
##
##clf=RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
##
##clf.fit(X,documents2)
##
##print(clf.predict(Y))
##
##print("---------------------SVD-SVM -------------")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

Y = vectorizer.transform(test)

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

X=svd.fit_transform(X)
print('Explained variance:\n', lsa.explained_variance_, '\n')
##Y = svd.transform(Y)
##
##
##clf=svm.SVC(gamma = 'scale')
##
##clf.fit(X,documents2)
##
##print(clf.predict(Y))

print("---------------------W2V -------------")
##def preprocess(text):
##    text = re.sub(r'[^\w\s]','',text)
##    tokens = text.lower()
##    tokens = tokens.split()
##    return tokens
listX=[]
for rowInd, row in documents.iterrows():
    listX.append(row['Content'].split())

listY=[]
for rowInd, row in documents2.iterrows():
    listY.append(row['Category'].split())

listTest=[]
for rowInd, row in documents2.iterrows():
    listTest.append(row['Content'].split())


##texts = [[text for text in doc.split()] for doc in documents]
##
##dictionary = corpora.Dictionary(texts)
##
##mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]
##
##vectorizer = TfidfVectorizer()
##vectorizer.fit(documents)
##X = vectorizer.transform(documents)
##y = documents2
##s = vectorizer.transform([test])
##s2 = vectorizer.transform([test2])
##clf = svm.SVC(gamma='scale')
##pipeline = Pipeline([('bow', mycorpus), ('classifier', clf)])
##pipeline.fit(documents, documents2)
##print(pipeline.score(documents, documents2))
##print(clf.predict(s2))

