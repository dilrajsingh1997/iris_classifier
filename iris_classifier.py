import random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import decomposition
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import numpy as np
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

from sklearn.metrics import accuracy_score

def get_accuracy(predictions, actual):
    correctness = (predictions==actual).sum()
    total = actual.size
    return float(correctness)/float(total)

data = pd.read_csv('iris-data-clean.csv').dropna()
dict_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['class'] = data['class'].map(dict_map)
#print data.head()

accuracy = {}
accuracy['DecisionTreeClassifier'] = 0
accuracy['RandomForestClassifier'] = 0
accuracy['LinearSVM'] = 0
accuracy['2ndDegreeSVM'] = 0
accuracy['3rdDegreeSVM'] = 0
accuracy['4thDegreeSVM'] = 0
accuracy['5thDegreeSVM'] = 0
accuracy['KNeighborsClassifier'] = 0
accuracy['NaiveBayesMultinomialNB'] = 0
accuracy['Logisticregression'] = 0

for _ in range(100):

    t = random.uniform(0.1, 0.4)
    train, test = train_test_split(data, test_size=t)

    feature_vector = data.columns[0:4]
    #print feature_vector

    feature_data = train[feature_vector].dropna()
    #print feature_data.head()

    response_data = train['class'].dropna()
    #print response_data.head()

    #from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    fitter_decision_tree = clf.fit(feature_data, response_data)
    predictions_decision_tree = fitter_decision_tree.predict(test[feature_vector])
    #print get_accuracy(predictions_decision_tree, test['class'])
    accuracy['DecisionTreeClassifier'] += get_accuracy(predictions_decision_tree, test['class'])

    #from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200)
    fitter_random_forest_classifier = clf.fit(feature_data, response_data)
    predictions_random_forest_classifier = fitter_random_forest_classifier.predict(test[feature_vector])
    #print get_accuracy(predictions_random_forest_classifier, test['class'])
    accuracy['RandomForestClassifier'] += get_accuracy(predictions_random_forest_classifier, test['class'])

    #from sklearn import svm
    clf = svm.SVC(kernel='linear', C=1)
    fitter_linear_svm = clf.fit(feature_data, response_data)
    predictions_linear_svm = fitter_linear_svm.predict(test[feature_vector])
    #print get_accuracy(predictions_linear_svm, test['class'])
    accuracy['LinearSVM'] += get_accuracy(predictions_linear_svm, test['class'])

    #from sklearn import svm
    clf = svm.SVC(kernel='poly', degree=2, C=1)
    fitter_poly_2_svm = clf.fit(feature_data, response_data)
    predictions_poly_2_svm = fitter_poly_2_svm.predict(test[feature_vector])
    #print get_accuracy(predictions_poly_2_svm, test['class'])
    accuracy['2ndDegreeSVM'] += get_accuracy(predictions_poly_2_svm, test['class'])

    clf = svm.SVC(kernel='poly', degree=3, C=1)
    fitter_poly_3_svm = clf.fit(feature_data, response_data)
    predictions_poly_3_svm = fitter_poly_3_svm.predict(test[feature_vector])
    #print get_accuracy(predictions_poly_3_svm, test['class'])
    accuracy['3rdDegreeSVM'] += get_accuracy(predictions_poly_3_svm, test['class'])

    clf = svm.SVC(kernel='poly', degree=4, C=1)
    fitter_poly_4_svm = clf.fit(feature_data, response_data)
    predictions_poly_4_svm = fitter_poly_4_svm.predict(test[feature_vector])
    #print get_accuracy(predictions_poly_4_svm, test['class'])
    accuracy['4thDegreeSVM'] += get_accuracy(predictions_poly_4_svm, test['class'])

    clf = svm.SVC(kernel='poly', degree=5, C=1)
    fitter_poly_5_svm = clf.fit(feature_data, response_data)
    predictions_poly_5_svm = fitter_poly_5_svm.predict(test[feature_vector])
    #print get_accuracy(predictions_poly_5_svm, test['class'])
    accuracy['5thDegreeSVM'] += get_accuracy(predictions_poly_5_svm, test['class'])

    #from sklearn import neighbours
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    fitter_k_nearest_neighbours = clf.fit(feature_data, response_data)
    predictions_k_nearest_neighbours = fitter_k_nearest_neighbours.predict(test[feature_vector])
    #print get_accuracy(predictions_k_nearest_neighbours, test['class'])
    accuracy['KNeighborsClassifier'] += get_accuracy(predictions_k_nearest_neighbours, test['class'])

    #from sklearn import naive_bayes
    clf = naive_bayes.MultinomialNB()
    fitter_naive_bayes = clf.fit(feature_data, response_data)
    predictions_naive_bayes = fitter_naive_bayes.predict(test[feature_vector])
    #print get_accuracy(predictions_naive_bayes, test['class'])
    accuracy['NaiveBayesMultinomialNB'] += get_accuracy(predictions_naive_bayes, test['class'])

    '''
    #from sklearn import decomposition
    pca = decomposition.PCA(n_components=3, whiten=True).fit(feature_data)
    feature_data_pca_3 = pca.transform(feature_data)
    print sum(pca.explained_variance_ratio_)
    
    pca = decomposition.PCA(n_components=2, whiten=True).fit(feature_data)
    feature_data_pca_2 = pca.transform(feature_data)
    print sum(pca.explained_variance_ratio_)
    
    pca = decomposition.PCA(n_components=1, whiten=True).fit(feature_data)
    feature_data_pca_1 = pca.transform(feature_data)
    print sum(pca.explained_variance_ratio_)
    '''

    #from sklearn import linear_model
    clf = linear_model.LogisticRegression()
    fitter_logistic_regression = clf.fit(feature_data, response_data)
    predictions_logistic_regression = fitter_logistic_regression.predict(test[feature_vector])
    #print get_accuracy(predictions_logistic_regression, test['class'])
    accuracy['Logisticregression'] += get_accuracy(predictions_logistic_regression, test['class'])


accuracy['DecisionTreeClassifier'] /= 100
accuracy['RandomForestClassifier'] /= 100
accuracy['LinearSVM'] /= 100
accuracy['2ndDegreeSVM'] /= 100
accuracy['3rdDegreeSVM'] /= 100
accuracy['4thDegreeSVM'] /= 100
accuracy['5thDegreeSVM'] /= 100
accuracy['KNeighborsClassifier'] /= 100
accuracy['NaiveBayesMultinomialNB'] /= 100
accuracy['Logisticregression'] /= 100

print(accuracy)

#plt.scatter([0, 1, 2, 3, 4, 5, 6, 7,8 ,9], accuracy.values())
plt.bar(range(len(accuracy)), accuracy.values(), align='center')
plt.xticks(range(len(accuracy)), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#plt.legend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], accuracy.keys())
plt.show()