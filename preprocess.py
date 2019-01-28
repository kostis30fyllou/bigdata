import numpy as np
import pandas as pd
from os import path
from PIL import Image 
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Read Data functions
def read_csv(path):
    df = pd.read_csv(path, sep='\t', index_col=0)
    return df

#WordCloud section functions
def getCategoryContents(df, category):
    text = " ".join(review for review in df[df["Category"] == category].Content)
    return text

def saveWordcloud(data, category):
    # Create and generate a word cloud image:
    print('Create a wordcloud for category', category)
    wordcloud=WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordclouds/"+category+".png", format="png")
    
#Duplicate section functions
def categoryDict(df, category):
    dict = []
    for content in df[df["Category"] == category].Content:
        dict.append(content)        
    return dict
    
def dictToVect(dict):
    vectorizer = HashingVectorizer(n_features=2**6)
    X = vectorizer.fit_transform(dict)
    return X.toarray()
    
def getIdsByCategory(df, category):
    return df[df["Category"] == category].Id.unique()
    

def getDuplicates(theta, categories, df):
    duplicates = []
    for category in categories:
        dict = categoryDict(df, category)
        X = dictToVect(dict)
        ids = []
        ids = getIdsByCategory(df, category)
        for i,id1 in enumerate(ids):
            vec1 = X[i].reshape(1,-1)
            for j,id2 in enumerate(ids[i+1:], start=i+1):
                vec2 = X[j].reshape(1,-1)
                cs = cosine_similarity(vec1, vec2)
                if cs[0][0]>= theta:
                    duplicates.append([id1, id2, cs[0][0]])
    return duplicates
    