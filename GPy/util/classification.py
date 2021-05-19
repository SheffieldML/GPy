# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np


def conf_matrix(p, labels, names=["1", "0"], threshold=0.5, show=True):
    """
    Returns error rate and true/false positives in a binary classification problem
    - Actual classes are displayed by column.
    - Predicted classes are displayed by row.

    :param p: array of class '1' probabilities.
    :param labels: array of actual classes.
    :param names: list of class names, defaults to ['1','0'].
    :param threshold: probability value used to decide the class.
    :param show: whether the matrix should be shown or not
    :type show: False|True
    """
    assert p.size == labels.size, "Arrays p and labels have different dimensions."
    decision = np.ones((labels.size, 1))
    decision[p < threshold] = 0
    diff = decision - labels
    false_0 = diff[diff == -1].size
    false_1 = diff[diff == 1].size
    true_1 = np.sum(decision[diff == 0])
    true_0 = labels.size - true_1 - false_0 - false_1
    error = (false_1 + false_0) / np.float(labels.size)
    if show:
        print(100.0 - error * 100, "% instances correctly classified")
        print("%-10s|  %-10s|  %-10s| " % ("", names[0], names[1]))
        print("----------|------------|------------|")
        print("%-10s|  %-10s|  %-10s| " % (names[0], true_1, false_0))
        print("%-10s|  %-10s|  %-10s| " % (names[1], false_1, true_0))
    return error, true_1, false_1, true_0, false_0
