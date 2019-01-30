import pandas as pd
from preprocess import *
 
df = read_csv("train_set.csv")
categories = df.Category.unique()
for category in categories:
    saveWordCloud(df, category)
