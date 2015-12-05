import sys
import numpy as np

from sklearn.externals.joblib import load

import names


def read_file(path):
    print 'Reading file...'
    fin = file(path)
    text = fin.read()
    fin.close()
    print 'Done.'
    return text


def predict(text, model_name, category_type):
    print 'Loading classifier...'
    clf = load('dumps/' + category_type + '_' + model_name + '_classifier.pkl')
    print 'Done.'

    print 'Predicting...'
    pred = clf.predict([text])[0]

    if model_name in ['linreg', 'svr']:
        pred = int(np.clip(np.round(pred), 0, 7))

    print names.categories[category_type][pred]


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python predict.py <nb|svc|logreg|svr|linreg> <stars|binary> <path to text>'
        sys.exit(0)

    if (sys.argv[2], sys.argv[1]) not in names.classifiers:
        print 'Error: Infeasible combination of targets and model'
        sys.exit(0)

    text = read_file(sys.argv[3])
    predict(text, sys.argv[1], sys.argv[2])