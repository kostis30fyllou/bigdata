import pandas as pd
from utils import *
from sklearn.metrics.pairwise import cosine_similarity

def getDuplicates(candidates, theta, df):
    duplicates = []
    vectorizer = toVectorizer(df)
    for ((i, docid1), (j, docid2)) in candidates:
        vec1, vec2 = toArray(df.iloc[i].Content, df.iloc[j].Content, vectorizer)
        cs = cosine_similarity(vec1, vec2)
        if cs[0][0]>= theta:
            duplicates.append({'Document_Id1': docid1, 'Document_Id2': docid2, 'Similarity': cs[0][0]})
    return duplicates
    

theta = float(input("Enter theta:"))
df = read_csv("train_set.csv")
candidates = getCandidates(df, char_ngram=5, seeds=100, bands=20, hashbytes=4)
duplicates = getDuplicates(candidates, theta, df);
result = pd.DataFrame(duplicates)
if theta == 0.7:	
    write_csv("duplicatePairs.csv", result)
else:
    print(result)

    
    

















