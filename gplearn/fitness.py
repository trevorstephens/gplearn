"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np

from scipy.stats import rankdata


def weighted_pearson(x1, x2, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        x1_demean = x1 - np.average(x1, weights=w)
        x2_demean = x2 - np.average(x2, weights=w)
        corr = ((np.sum(w * x1_demean * x2_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * x1_demean ** 2) *
                         np.sum(w * x2_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0


def weighted_spearman(x1, x2, w):
    """Calculate the weighted Spearman correlation coefficient."""
    x1_ranked = np.apply_along_axis(rankdata, 0, x1)
    x2_ranked = np.apply_along_axis(rankdata, 0, x2)
    return weighted_pearson(x1_ranked, x2_ranked, w)
