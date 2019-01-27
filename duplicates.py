import pandas as pd
from preprocess import *

df = read_csv("train_set.csv").head(100)
categories = df.Category.unique()
duplicates = getDuplicates(0.7, categories, df)
print(duplicates)

    
    

















