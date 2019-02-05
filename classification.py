from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def svm():
    print("Create a SVM.SVC classifier")
    clf = SVC(gamma='scale')
    return clf

def randomForest():
    print("Create a Random Forest classifier")
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    return clf


