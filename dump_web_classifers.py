import sys

from sklearn.externals.joblib import dump

import data_loader
import names
import tfidf_pipeline
import model_presets


if __name__ == '__main__':
    for (category_name, model_name) in [('stars', 'linreg'), ('binary', 'svc')]:

        print 'Loading ' + category_name + ' data'
        train,_ = data_loader.load('split', names.categories[category_name])

        print 'Training ' + model_name
        clf = tfidf_pipeline.make(model_name)
        clf.fit(train.data, train.target)

        print 'Dumping ' + model_name
        dump(clf, 'web_clf_' + category_name + '.pkl')
        