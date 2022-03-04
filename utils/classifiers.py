from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tabulate import tabulate


def classify_SVM(_X_train, _X_test, _y_train, _y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("SVM Accuracy:", acc)
    return acc


def classify_DTC(_X_train, _X_test, _y_train, _y_test):
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("DT Accuracy:", acc)
    return acc


def classify_EDTC(_X_train, _X_test, _y_train, _y_test):
    clf = ExtraTreesClassifier(random_state=0)
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("ET Accuracy:", acc)
    return acc


def classify_LDA(_X_train, _X_test, _y_train, _y_test):
    """Linear Discriminant Analysis"""
    clf = LinearDiscriminantAnalysis()
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("LDA Accuracy:", acc)
    return acc


def classify_KNN(_X_train, _X_test, _y_train, _y_test):
    """k-nearest neighbors"""
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("KNN Accuracy:", acc)
    return acc


def classify_GNB(_X_train, _X_test, _y_train, _y_test):
    """Gaussian Naive Bayes"""
    gnb = GaussianNB()
    y_pred = gnb.fit(_X_train, _y_train).predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("GNB Accuracy:", acc)
    return acc


def classify_MLP(_X_train, _X_test, _y_train, _y_test):
    """Multi-layer Perceptron classifier"""
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("MLP Accuracy:", acc)
    return acc


def classify_RFC(_X_train, _X_test, _y_train, _y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    acc = metrics.accuracy_score(_y_test, y_pred)
    # print("RFC Accuracy:", acc)
    return acc


def classify(X_train, X_test, y_train, y_test, results=None, fs='None'):
    result = [
        ['SVM', classify_SVM(X_train, X_test, y_train, y_test)],
        ['Decision Tree', classify_DTC(X_train, X_test, y_train, y_test)],
        ['Extra Trees', classify_EDTC(X_train, X_test, y_train, y_test)],
        ['Linear Discriminant Analysis', classify_LDA(X_train, X_test, y_train, y_test)],
        ['kNN', classify_KNN(X_train, X_test, y_train, y_test)],
        ['Gaussian Naive Bayes', classify_GNB(X_train, X_test, y_train, y_test)],
        ['Multi-layer Perceptron', classify_MLP(X_train, X_test, y_train, y_test)],
        ['Random Forest', classify_RFC(X_train, X_test, y_train, y_test)],
    ]
    if results is not None:
        results.append([fs] + [x[1] for x in result])
    print(tabulate(result, tablefmt='fancy_grid'))
