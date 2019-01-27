import numpy as np
import pandas as pd
from os import path
from PIL import Image 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

def read_csv(path):
	df = pd.read_csv(path, sep='\t', index_col=0)
	return df

def getCategoryContents(df, category):
	text = " ".join(review for review in df[df["Category"] == category].Content)
	return text

def categoryDict(df, category):
    dict = []
    for content in df[df["Category"] == category].Content:
        dict.append(content)        
    return dict
        

def saveWordcloud(data, category):
	# Create and generate a word cloud image:
	wordcloud=WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data)

	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig("wordclouds/"+category+".png", format="png")

