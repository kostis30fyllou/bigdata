from utils import *
from sklearn.metrics import *
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score
from classification import *
from features import *

def evaluationMetrics(documents, categories):
    metrics = []
    for (X,feature) in ((bow(documents), 'BoW'), 
                        (svd(documents), 'SVD'), 
                        (w2v(documents), 'W2V')):
        for (clf, name) in ((svm(), 'SVM'),
                            (randomForest(), 'Random Forest')):
            print('*' * 80)
            print("Executing", name, "with feature", feature)
            print('*' * 80)                   
            scoring = {'accuracy': 'accuracy',
                       'precision': 'precision_weighted',
                       'recall': 'recall_weighted'}
            scores = cross_validate(clf, X, categories, cv=10 , scoring=scoring, n_jobs=-1, return_train_score=True)
            accuracy = scores['test_accuracy'].mean()
            precision = scores['test_precision'].mean()
            recall = scores['test_recall'].mean()
            F1 = 2 * (precision * recall) / (precision + recall)
            metrics.append([{name+'('+feature+')':"{0:.2f}".format(accuracy)}, {name+'('+feature+')':"{0:.2f}".format(precision)}, {name+'('+feature+')':"{0:.2f}".format(recall)}, {name+'('+feature+')':"{0:.2f}".format(F1)}])
    return metrics

def beatTheBenchmark(documents, categories, testDf):
    print('*' * 80)
    print("Executing Beat the Benchmark method")
    print('*' * 80)  
    X, Y = myfeature(documents, testDf.Content)
    clf = svm()
    scoring = {'accuracy': 'accuracy',
               'precision': 'precision_weighted',
               'recall': 'recall_weighted'}
    scores = cross_validate(clf, X, categories, cv=10 , scoring=scoring, n_jobs=-1, return_train_score=True)
    accuracy = scores['test_accuracy'].mean()
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    F1 = 2 * (precision * recall) / (precision + recall)
    clf.fit(X, categories)
    predicted = clf.predict(Y)
    return ([{'MyMethod':"{0:.2f}".format(accuracy)}, {'MyMethod':"{0:.2f}".format(precision)}, 
	    {'MyMethod':"{0:.2f}".format(recall)}, {'MyMethod':"{0:.2f}".format(F1)}], 
            {'Test_Document_ID': testDf.Id.tolist(), 'Predicted_Category': predicted})
            
documents = read_csv("train_set.csv").Content
categories = read_csv("train_set.csv").Category
testDocuments = read_csv("test_set.csv")
metrics = pd.DataFrame({'StatisticMeasure': ['Accuracy', 'Precision', 'Recall', 'F-Measure']})
for metric in evaluationMetrics(documents, categories):
    metrics = metrics.join(pd.DataFrame(metric))
mymethod,predicted = beatTheBenchmark(documents, categories, testDocuments)
metrics = metrics.join(pd.DataFrame(mymethod))
predicts = pd.DataFrame(predicted)
write_csv("EvaluationMetric_10fold.csv", metrics)
write_csv("testSet_categories.csv", predicts)
