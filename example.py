from sklearn.feature_extraction.text import HashingVectorizer
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

df = read_csv("train_set.csv")
dict = categoryDict(df, 'Business')
#print(corpus['Business'][0])

X = dictToVect(dict)

ids = getIdsByCategory(df, category)
for i in range(len(ids)):
    vec1 = X[i].reshape(1,-1)
    id1 = ids[i]
    for j in range(i+1, len(ids)):
        vec2 = X[j].reshape(1,-1)
        id2 = ids[j]
        cs = cosine_similarity(vec1, vec2)
        print(id1, id2, cs)
    
    
    
#print(cosine_similarity(X[3].reshape(1,-1),X[1].reshape(1,-1)))
