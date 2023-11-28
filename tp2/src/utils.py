import numpy as np
from numba import njit


@njit
def random_choice(prob: np.ndarray):
    """Workaround for a random choice function since Numba does not support the
    optional p argument (probabilities array) from numpy.random.choice

    https://github.com/numba/numba/issues/2539
    https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#permutations

    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return np.searchsorted(np.cumsum(prob), np.random.random(), side='right')
