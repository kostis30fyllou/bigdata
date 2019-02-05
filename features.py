from gensim import corpora
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from utils import *

def bow(documents):
    print('Trasform train data with bag of words')
    train = [[text for text in doc.split()] for doc in documents]
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in train]
    model = LdaTransformer(id2word=dictionary, num_topics=5, iterations=20)
    X = model.fit_transform(corpus)
    return X

def svd(documents):
    print('Trasform train data with SVD')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    svd = TruncatedSVD(n_components=600, n_iter=7, random_state=42)
    svd.fit(X)
    print("Variance with this number of components is %1.2f" % getVariance(svd.explained_variance_ratio_))
    X = svd.transform(X)
    return X

def w2v(documents):
    print('Trasform train data with average word to vector')
    trains = [[text for text in doc.split()] for doc in documents]
    model = word2vec.Word2Vec(documents,\
                          workers=4,\
                          size=100,\
                          min_count=1)
    X = getAvgFeatureVecs(trains, model, 100)
    return X

def myfeature(documents, testDocuments):
    print('Trasform train data with my feature')
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(documents)
    Y = vectorizer.transform(testDocuments)
    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    svd.fit(X)
    print("Variance with this number of components is %1.2f" % getVariance(svd.explained_variance_ratio_))
    X = svd.transform(X)
    Y = svd.transform(Y)
    return X,Y
