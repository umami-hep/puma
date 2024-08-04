from __future__ import annotations

import numpy as np

from puma.utils.precision_score import precision_score_per_class
from puma.utils.recall_score import recall_score_per_class

# Sample size
N = 100

# Number of target classes
Nclass = 3

# Dummy target labels
targets = np.random.randint(0, Nclass, size=N)
# Making sure that there is at least one sample for each class
targets = np.append(targets, np.array(list(range(Nclass))))

# Dummy predicted labels
predictions = np.random.randint(0, Nclass, size=(N + Nclass))


# Unweighted precision and recall
uw_precision = precision_score_per_class(targets, predictions)
uw_recall = recall_score_per_class(targets, predictions)
print("Unweighted case:")
print("Per-class precision:")
print(uw_precision)
print("Per-class recall:")
print(uw_recall)
print(" ")

# Weighted precision and recall
# Dummy sample weights
sample_weights = np.random.rand(N + Nclass)

uw_precision = precision_score_per_class(targets, predictions, sample_weights)
uw_recall = recall_score_per_class(targets, predictions, sample_weights)
print("Weighted case:")
print("Per-class precision:")
print(uw_precision)
print("Per-class recall:")
print(uw_recall)
print(" ")
