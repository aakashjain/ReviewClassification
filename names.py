categories = {'binary': ['neg','pos'],
              'stars' : ['01','02','03','04','07','08','09','10'],
              'all'   : ['01','02','03','04','07','08','09','10','neg','pos'],
             }

models = ['nb', 'svc', 'logreg', 'svr', 'linreg']

classifiers = [('binary', 'nb'), ('binary', 'svc'), ('binary', 'logreg'),
               ('stars', 'svr'), ('stars', 'linreg')]