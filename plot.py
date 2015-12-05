import numpy as np
import matplotlib.pyplot as plot

import names


def plot_confusion_matrix(cm, category_name, title='Confusion matrix'):
    categories = names.categories[category_name]

    cm = np.array(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Greys)
    plot.title(title)
    plot.colorbar()

    tick_marks = np.arange(len(categories))
    plot.xticks(tick_marks, categories, rotation=45)
    plot.yticks(tick_marks, categories)

    plot.tight_layout()
    plot.ylabel('True')
    plot.xlabel('Predicted')

    plot.show()