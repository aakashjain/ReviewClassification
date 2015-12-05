import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import StratifiedShuffleSplit


def load(dataset, categories):
    if dataset == 'full':
        train = load_files('aclImdb/aggregate/', categories=categories)
        return train

    elif dataset == 'split':    
        train = load_files('aclImdb/train/', categories=categories)
        test = load_files('aclImdb/test/', categories=categories)
        return (train, test)


def split(data, test_size):
    X, y = np.array(data.data), np.array(data.target)

    splitter = StratifiedShuffleSplit(n_iter=1, test_size=test_size)
    train, test = next(splitter.split(X, y))

    return X[train], y[train], X[test], y[test]