import pandas as pd
from utils import *
from sklearn.metrics.pairwise import cosine_similarity

def getDuplicates(candidates, theta, df):
    print('Finding duplicates from candidates')
    duplicates = []
    vectorizer = toVectorizer(df)
    for ((i, docid1), (j, docid2)) in candidates:
        vec1, vec2 = toArray(df.iloc[i].Content, df.iloc[j].Content, vectorizer)
        cs = cosine_similarity(vec1, vec2)
        if cs[0][0]>= theta:
            duplicates.append({'Document_Id1': docid1, 'Document_Id2': docid2, 'Similarity': cs[0][0]})
    print('It found', len(duplicates), 'duplicates')
    return duplicates
    

theta = float(input("Enter theta:"))
df = read_csv("train_set.csv")
candidates = getCandidates(df)
duplicates = getDuplicates(candidates, theta, df);
result = pd.DataFrame(duplicates)
if theta == 0.7:	
    write_csv("duplicatePairs.csv", result)
else:
    print(result)

    
    

















