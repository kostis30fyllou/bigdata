from utils import *
from sklearn.metrics import *
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from classification import *
from features import *

def evaluationMetrics(clf, documents, categories):
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    auc = 0.0
    kf = KFold(n_splits=10)
    classes = categories.unique()
    splits = kf.get_n_splits()
    for train_index, test_index in kf.split(documents):
        train = np.array(documents)[train_index]
        test = np.array(documents)[test_index]
        targetTrain = np.array(categories)[train_index]
        targetTest = binarize(np.array(categories)[test_index], classes)
        if clf == 'SVM(BoW)':
            X,Y = bow(train, test)
            predicted = svm(X, Y, targetTrain)
        elif clf == 'RandomForest(BoW)':
            X,Y = bow(train, test)
            predicted = randomForest(X, Y, targetTrain)
        elif clf == 'SVM(SVD)':
            X,Y = svd(train, test)
            predicted = svm(X, Y, targetTrain)
        elif clf == 'RandomForest(SVD)':
            X,Y = svd(train, test)
            predicted = randomForest(X, Y, targetTrain)
        elif clf == 'SVM(W2V)':
            X,Y = w2v(train, test)
            predicted = svm(X, Y, targetTrain)
        elif clf == 'RandomForest(W2V)':
            X,Y = w2v(train, test)
            predicted = randomForest(X, Y, targetTrain)
        predicted = binarize(predicted, classes)
        accuracy += accuracy_score(targetTest, predicted)
        precision += precision_score(targetTest, predicted, average='weighted')
        recall += recall_score(targetTest, predicted, average='weighted')
        auc += roc_auc_score(targetTest.ravel(), predicted.ravel(), average='weighted')
    accuracy = accuracy/splits
    precision = precision/splits
    recall = recall/splits
    F1 = 2 * (precision * recall) / (precision + recall)
    auc = auc/splits
    return [{clf:accuracy}, {clf:precision}, {clf:recall}, {clf:F1}, {clf:auc}]

documents = read_csv("train_set.csv").head(100).Content
categories = read_csv("train_set.csv").head(100).Category
df = pd.DataFrame([{'StatisticMeasure':'Accuracy'}, {'StatisticMeasure':'Precision'}, {'StatisticMeasure':'Recall'}, {'StatisticMeasure':'F-Measure'}, {'StatisticMeasure':'AUC'}])
#df = df.join(pd.DataFrame(evaluationMetrics('SVM(BoW)', documents, categories)))
#df = df.join(pd.DataFrame(evaluationMetrics('RandomForest(BoW)', documents, categories)))
#df = df.join(pd.DataFrame(evaluationMetrics('SVM(SVD)', documents, categories)))
#df = df.join(pd.DataFrame(evaluationMetrics('RandomForest(SVD)', documents, categories)))
df = df.join(pd.DataFrame(evaluationMetrics('SVM(W2V)', documents, categories)))
df = df.join(pd.DataFrame(evaluationMetrics('RandomForest(W2V)', documents, categories)))
print(df)
