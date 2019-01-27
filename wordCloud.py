import pandas as pd
from preprocess import *
 
df = read_csv("train_set.csv")

categories = df.groupby("Category")
for category in categories.Category:
	text = getCategoryContents(df, category)
	saveWordcloud(text, category)







        
