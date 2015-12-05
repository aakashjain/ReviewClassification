from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import model_presets


def __tfidf():
    return TfidfVectorizer(decode_error='replace', strip_accents='unicode',
                           stop_words='english', lowercase=True,
                           analyzer='word', ngram_range=(1,3),
                           use_idf=True, smooth_idf=True, sublinear_tf=True)


def make(model_name):
    return Pipeline([('tfidf', __tfidf()),
                     ('model', model_presets.get(model_name))])