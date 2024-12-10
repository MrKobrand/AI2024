import pandas as pd
from scipy.spatial.distance import cdist


def compute_minkowski_distances(X, p=3):
    return pd.DataFrame(cdist(X, X, metric='minkowski', p=p))


def compute_euclidean_distances(X):
    return pd.DataFrame(cdist(X, X, metric='euclidean'))


def compute_manhattan_distances(X):
    return pd.DataFrame(cdist(X, X, metric='cityblock'))


def compute_cosine_distances(X):
    return pd.DataFrame(cdist(X, X, metric='cosine'))


def compute_jensenshannon_distances(X):
    return pd.DataFrame(cdist(X, X, metric='jensenshannon'))
