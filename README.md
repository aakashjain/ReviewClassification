Classification of Film Reviews using Machine Learning
=====================================================

Mini project by Aakash Jain, Punit Chhajer

Instructions:
-------------

- Requires Python 2.7
- Install scikit-learn v0.18 or later along with all dependencies
- Install flask along with all dependencies
- Download the [ACL IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) dataset
- Extract aclImdb_v1.tar.gz to aclImdb/
- Run organize_dataset.py

For CLI app:
- Run dump_classifiers.py - This will take several hours and may make your computer unresponsive
- To train and test individual classifiers, run train.py as 
  `python train.py <nb|svc|logreg|svr|linreg> <stars|binary> <dump?>
- To predict with a dumped classifier, run predict.py as 
  `python predict.py <nb|svc|logreg|svr|linreg> <stars|binary> <path to file>

For webapp:
- Run dump_web_classifiers.py and put the two new dumps into webapp/
- Start the server with webapp/main.py
- Browse to 127.0.0.1:5000
