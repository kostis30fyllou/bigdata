from gensim import corpora
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.sklearn_api import W2VTransformer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from utils import *

def bow(documents, testDocuments):
    train = [[text for text in doc.split()] for doc in documents]
    test = [[text for text in doc.split()] for doc in testDocuments]
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in train]
    testCorpus = [dictionary.doc2bow(doc) for doc in test]
    model = LdaTransformer(id2word=dictionary, num_topics=5, iterations=20)
    X = model.fit_transform(corpus)
    Y = model.transform(testCorpus)
    return X,Y

def svd(documents, testDocuments):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    Y = vectorizer.transform(testDocuments)
    svd = TruncatedSVD(n_components=600, n_iter=7, random_state=42)
    svd.fit(X)
    print("Variance with this number of components is %1.2f" % getVariance(svd.explained_variance_ratio_))
    X = svd.transform(X)
    Y = svd.transform(Y)
    return X,Y

def w2v(documents, testDocuments):
    train = [[text for text in doc.split()] for doc in documents]
    test = [[text for text in doc.split()] for doc in testDocuments]
    model = W2VTransformer(size=10, min_count=1)
    X = model.fit(train).transform(train[0])
    Y = model.transform(test[0])
    return X,Y
    
