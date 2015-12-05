import train
import names


if __name__ == '__main__':
    for (cat, model) in names.classifiers:
        train.train(model, cat, dump=True)