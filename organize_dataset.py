from os import listdir, makedirs, sep
from os.path import abspath, join, exists
from shutil import copy

from sklearn.externals.joblib import Parallel, delayed

import names


def __categorize_by_stars(path):
    path = abspath(path)

    for i in names.categories['stars']:
        stars_dir = join(path, i)
        if not exists(stars_dir):
            makedirs(stars_dir)

    for cat in ['pos','neg']:
        for f in listdir(join(path, cat)):
            stars = f.split('.')[0].split('_')[1].rjust(2,'0') + sep
            copy(join(path, cat, f), join(path, stars))


def categorize_by_stars(*paths):
    Parallel(n_jobs=-1)(delayed(__categorize_by_stars)(path)
                        for path in paths)


def __create_aggregate_set(cat, path):
    n = 0

    for data in ['train','test']:
        src = join(path, data, cat)
        if not exists(src):
            continue

        dest = join(path, 'aggregate', cat)
        if not exists(dest):
            makedirs(dest)

        for f in listdir(src):
            copy(join(src, f), join(dest, str(n) + '.txt'))
            n += 1


def create_aggregate_set(path):
    path = abspath(path)
    Parallel(n_jobs=-1)(delayed(__create_aggregate_set)(cat, path)
                        for cat in names.categories['all'])


if __name__ == '__main__':
    categorize_by_stars('aclImdb/train', 'aclImdb/test')
    create_aggregate_set('aclImdb/')