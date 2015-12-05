from flask import Flask, render_template, redirect, url_for, request
from sklearn.externals.joblib import load
import numpy as np


app = Flask(__name__)

labels = {'stars': ['1','2','3','4','7','8','9','10'],
          'binary': ['neg','pos']}

models = {'stars': load('web_clf_stars.pkl'),
          'binary': load('web_clf_binary.pkl')}


@app.route('/', methods=['GET', 'POST'])
def main():
    answer = None
    typ = 'stars'
    X = ''
    if request.method == 'POST':
        X = request.form['comments']
        typ = request.form['type']
        y = models[typ].predict([X])[0]
        y = int(np.clip(np.round(y), 0, 7))
        answer = labels[typ][y]
    return render_template('index.html', answer=answer, X=X, typ=typ)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)