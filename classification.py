from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def svm(X, Y, categories):
    clf = SVC(gamma='scale')
    clf.fit(X, categories)
    predicted = clf.predict(Y)
    return predicted

def randomForest(X, Y, categories):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(X, categories)
    predicted = clf.predict(Y)
    return predicted


