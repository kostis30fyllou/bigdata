import gensim
from gensim import corpora
from pprint import pprint
from utils import *
from sklearn import svm
import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from gensim.sklearn_api.ldamodel import LdaTransformer
 
from sklearn.pipeline import Pipeline

documents = read_csv("train_set.csv").head(1000).Content
categories = read_csv("train_set.csv").head(1000).Category
test = read_csv("test_set.csv").iloc[0].Content
#test2 = read_csv("test_set.csv").iloc[1].Content

texts = [[text for text in doc.split()] for doc in documents]

dictionary = corpora.Dictionary(texts)

mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]
test = dictionary.doc2bow([test])
model = LdaTransformer(id2word=dictionary, num_topics=5, iterations=20)
model.fit(mycorpus)
X = model.transform(mycorpus)
#vectorizer = TfidfVectorizer()
#vectorizer.fit(documents)
#X = vectorizer.transform(documents)
#y = documents2
s = model.transform([test])
#s2 = vectorizer.transform([test2])
clf = svm.SVC(gamma='scale')
clf.fit(X, categories)
print(clf.predict(s))
#print(clf.predict(s2))

