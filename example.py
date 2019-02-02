import gensim
from gensim import corpora
from pprint import pprint
from utils import *
from sklearn import svm
import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
 
from sklearn.pipeline import Pipeline

documents = read_csv("train_set.csv").head(100).Content
documents2 = read_csv("train_set.csv").head(100).Category
test = read_csv("test_set.csv").iloc[0].Content
#test2 = read_csv("test_set.csv").iloc[1].Content

texts = [[text for text in doc.split()] for doc in documents]

dictionary = corpora.Dictionary(texts)

mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]

#vectorizer = TfidfVectorizer()
#vectorizer.fit(documents)
#X = vectorizer.transform(documents)
#y = documents2
#s = vectorizer.transform([test])
#s2 = vectorizer.transform([test2])
clf = svm.SVC(gamma='scale')
pipeline = Pipeline([('bow', mycorpus), ('classifier', clf)])
pipeline.fit(documents, documents2)
print(pipeline.score(documents, documents2))
#print(clf.predict(s2))

