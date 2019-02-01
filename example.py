import gensim
from gensim import corpora
from pprint import pprint
from utils import *
from sklearn import svm

documents = read_csv("train_set.csv").Content
texts = [[text for text in doc.split()] for doc in documents]
dictionary = corpora.Dictionary()
mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]
clf = svm.SVC(gamma='scale')
clf.fit(mycorpus[0],mycorpus[1])
print(clf.score(mycorpus[1]))

