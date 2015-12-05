import sys
from os import makedirs
from os.path import exists, join
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib

import data_loader
import tfidf_pipeline
import names
import scorers
import plot


def train(model_name, category_type, dump=False):
    clf = tfidf_pipeline.make(model_name)

    categories = names.categories[category_type]

    print 'Loading data...'
    data = data_loader.load('full', categories)
    train_X, train_y, test_X, test_y = data_loader.split(data, 0.1)
    print 'Done.'

    print 'Training...'
    clf.fit(train_X, train_y)
    print 'Done.'

    print 'Testing...'
    predicted = clf.predict(test_X)

    if model_name in ['svr', 'linreg']:
        predicted = np.clip(np.round(predicted), 0, 7)
        accuracy = scorers.err1(test_y, predicted)
        print 'Off-by-one accuracy: ' +  str(accuracy)
    else:
        accuracy = scorers.err0(test_y, predicted)
        print 'Exact accuracy: ' +  str(accuracy)
        print classification_report(test_y, predicted, target_names=categories)
    cm = confusion_matrix(test_y, predicted)
    print cm
    plot.plot_confusion_matrix(cm, category_type)

    if dump:
        print 'Saving classifier...'
        if not exists('dumps'):
            makedirs('dumps')
        joblib.dump(clf, join('dumps', category_type + '_' + model_name + '_classifier.pkl'))
        print 'Done.'

    return clf


if __name__ == '__main__':
    if len(sys.argv) not in (3,4):
        print 'Usage: python train.py <nb|svc|logreg|svr|linreg> <stars|binary> <dump?>'
        sys.exit(0)

    if (sys.argv[2], sys.argv[1]) not in names.classifiers:
        print 'Error: Infeasible combination of targets and model'
        sys.exit(0)

    dump = len(sys.argv) == 4 and sys.argv[3] == 'dump'
    train(sys.argv[1], sys.argv[2], dump)