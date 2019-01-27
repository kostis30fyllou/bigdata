##from LocalitySensitiveHashing import *
##
##datafile = "train_set.csv"
##lsh = LocalitySensitiveHashing(
##                   datafile = datafile,
##                   dim = 10,
##                   r = 50,
##                   b = 100,
##                   expected_num_of_clusters = 10,
##          )
##print(lsh)
##
##
##lsh.get_data_from_csv()
##lsh.initialize_hash_store()
##lsh.hash_all_data()
##similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
##coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
##merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )
##lsh.write_clusters_to_file( merged_similarity_groups, "clusters.txt" )
###https://learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
from preprocess import *
import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens


##Kathe Content to exw kanei mia lista apo lekseis

myDict=wordCloudCategory('Business')
myList=[]
k=0
for i in myDict['Business']:
    myList.append(preprocess(myDict['Business'][k]))
    k=k+1
    print('The shingles (tokens) are:', preprocess(myDict['Business'][k]))

    
    

















