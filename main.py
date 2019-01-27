import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import itertools
import re, math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer , CountVectorizer
from sklearn.neighbors import LSHForest

def read_csv(path):
	df = pd.read_csv(path, sep='\t')
	return df

def getCategoryContents(df, category):
	text = " ".join(review for review in df[df["Category"] == category].Content)
	return text

def getContent(category, i):
    text = df[df["Category"] == category].Content[i]
    return text

def saveWordcloud(data, category):
	# Create and generate a word cloud image:
	wordcloud=WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data)

	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig("wordclouds/"+category+".png", format="png")
	


df = read_csv("train_set.csv")

#categories = df.groupby("Category").head()
#for category in categories.Category:
	#text = getCategoryContents(df, category)
	#save_wordcloud(text, category)

mylist1 = []
mylist1.append(getContent("Business", 0))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mylist1)
	
mylist2 = []
mylist2.append(getContent("Business", 1))

Y = vectorizer.fit_transform(mylist2)

lshf=LSHForest(random_state=42)
lshf.fit(X.toarray())
distances,indices = lshf.kneighbors(Y,n_neighbors=2)
print(indices)
	



