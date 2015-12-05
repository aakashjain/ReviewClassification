import numpy as np

from sklearn.metrics import make_scorer


def err0(y, pred):
    pred2 = np.clip(np.round(pred), 0, 7)
    return np.mean(np.abs(pred2 - y) == np.zeros(len(y)))


def err1(y, pred):
    pred2 = np.clip(np.round(pred), 0, 7)
    return np.mean(np.abs(pred2 - y) <= np.ones(len(y)))


def err2(y, pred):
    pred2 = np.clip(np.round(pred), 0, 7)
    return np.mean(np.abs(pred2 - y) <= 2 * np.ones(len(y)))


err0_scorer = make_scorer(err0)
err1_scorer = make_scorer(err1)
err2_scorer = make_scorer(err2)