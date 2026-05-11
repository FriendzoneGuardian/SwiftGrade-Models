import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Short_Answer_NLP/src')))
from metrics import compute_qwk
from sklearn.metrics import cohen_kappa_score

y_true = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y_pred = np.array([2, 4, 4, 4, 6, 8, 8, 9, 9,  11, 10])

custom_qwk = compute_qwk(y_true, y_pred, 13)
sklearn_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

print(f"Custom QWK : {custom_qwk}")
print(f"Sklearn QWK: {sklearn_qwk}")
