import numpy as np
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from lsh import cache, minhash # https://github.com/mattilyra/lsh

def read_csv(path):
    df = pd.read_csv(path, sep='\t', index_col=0)
    return df

def write_csv(path, df):
    df.to_csv(path, sep='\t', index=False)

def getCategoryContents(df, category):
    text = " ".join(review for review in df[df["Category"] == category].Content)
    return text

def toVectorizer(df):
    dict = []
    for content in df.Content:
        dict.append(content)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(dict)        
    return vectorizer
    
def toArray(doc1, doc2, vectorizer):
    X = vectorizer.transform(doc1)
    Y = vectorizer.transform(doc2)
    return X, Y

def getCandidates(df, char_ngram=5, seeds=100, bands=20, hashbytes=4):
    print('Finding cadindate duplicate pairs with LSH technique')
    sims = []
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))
    lshcache = cache.Cache(num_bands=bands, hasher=hasher)
    for i in range(0, len(df.Id)):
        docid = df.iloc[i].Id
        content = df.iloc[i].Content
        fingerprint = hasher.fingerprint(content.encode('utf8'))
        # in addition to storing the fingerpring store the line
        # number and document ID to help analysis later on
        lshcache.add_fingerprint(fingerprint, doc_id=(i, docid))
    candidatePairs = set()
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) > 1:            
                pairs = set(itertools.combinations(b[bucket_id], r=2))
                candidatePairs.update(pairs)
    return candidatePairs
	