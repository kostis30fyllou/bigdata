from sklearn.feature_extraction.text import HashingVectorizer , CountVectorizer
from sklearn.neighbors import LSHForest
from preprocess import *
import re
from sklearn.metrics.pairwise import cosine_similarity

##dict=wordCloudCategory('Business')
##
##corpus = dict['Business']
##vectorizer = HashingVectorizer()
##X = vectorizer.fit_transform(corpus)
##print(X.toarray()) 
##
##
##
##
##
##
##lshf=LSHForest(random_state=42)
##lshf.fit(X.toarray())
##distances,indices = lshf.kneighbors(X.toarray(), n_neighbors=2)
##print(indices)


corpus = wordCloudCategory('Business')
#print(corpus['Business'][0])

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus['Business'])
X = X.toarray()
print(cosine_similarity(X[3].reshape(1,-1),X[1].reshape(1,-1)))
