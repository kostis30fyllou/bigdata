import gensim
from gensim import corpora
from pprint import pprint
from utils import *
from sklearn import svm
import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from gensim.sklearn_api.ldamodel import LdaTransformer
from features import *
 
from sklearn.pipeline import Pipeline

documents = read_csv("train_set.csv").head(1000).Content
categories = read_csv("train_set.csv").head(1000).Category
test = read_csv("test_set.csv").iloc[0].Content
#print(binarize(categories, read_csv("train_set.csv").Category.unique()))
X,Y = w2v(documents, categories)

