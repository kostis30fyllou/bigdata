import pandas as pd
from preprocess import *

theta = float(input("Enter theta:"))
df = read_csv("train_set.csv")
duplicates = getDuplicates(theta, df)

    
    

















