from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression


def get(model_name):
    return {'nb': MultinomialNB(alpha=0.1),

            'svc': LinearSVC(loss='hinge', penalty='l2', C=2.0, 
                             multi_class='ovr', dual=True),

            'logreg': LogisticRegression(multi_class='multinomial',
                                         solver='newton-cg', penalty='l2',
                                         C=100, dual=False, n_jobs=-1),

            'linreg': LinearRegression(n_jobs=-1),

            'svr': LinearSVR(epsilon=0.001, C=50, dual=True,
                             loss='epsilon_insensitive'),
           
           }[model_name]