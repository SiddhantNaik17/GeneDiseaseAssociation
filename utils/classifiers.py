from joblib import dump
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tabulate import tabulate

classifiers = {
    'SVM': svm.SVC(kernel='linear'),
    'DT': DecisionTreeClassifier(criterion='entropy', random_state=0),
    'ET': ExtraTreesClassifier(random_state=0),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'GNB': GaussianNB(),
    'MLP': MLPClassifier(alpha=1e-5, hidden_layer_sizes=(15,), random_state=1),
    'RF': RandomForestClassifier(max_depth=2, random_state=0)
}


def _classify(x_train, x_test, y_train, y_test, classifier, dump_name=None):
    classifier.fit(x_train, y_train)
    if dump_name:
        dump(classifier, dump_name)
    y_pred = classifier.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)


def classify(x_train, x_test, y_train, y_test, results=None, fs='None', dump_name=None):
    result = [
        ['SVM',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['SVM'])],
        ['Decision Tree',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['DT'])],
        ['Extra Trees',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['ET'])],
        ['Linear Discriminant Analysis',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['LDA'],
                   dump_name=dump_name)],
        ['kNN',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['KNN'])],
        ['Gaussian Naive Bayes',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['GNB'])],
        ['Multi-layer Perceptron',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['MLP'])],
        ['Random Forest',
         _classify(x_train, x_test, y_train, y_test,
                   classifier=classifiers['RF'])],
    ]

    if results is not None:
        results.append([fs] + [x[1] for x in result])

    print(tabulate(result, tablefmt='fancy_grid'))
