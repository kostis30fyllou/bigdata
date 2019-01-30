import pandas as pd
from preprocess import *

theta = float(input("Enter theta:"))
df = read_csv("train_set.csv")
categories = df.Category.unique()
duplicates = getDuplicates(theta, categories, df)
if theta == 0.7:	
    write_csv("out.csv", duplicates)
else:
    print(duplicates)

    
    

















