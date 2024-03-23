from __future__ import annotations

import numpy as np

from puma.utils.confusion_matrix import confusion_matrix

# Sample size
N = 100

# Number of target classes
Nclass = 3

# Dummy target labels
targets = np.random.randint(0, Nclass + 1, size=N)

# Dummy predicted labels
predictions = np.random.randint(0, Nclass + 1, size=N)


# Confusion matrix examples:

# Unweighted confusion matrix, normalized on all entries
unweighted_cm = confusion_matrix(targets, predictions)
print("Unweighted, normalized on all entries, CM:")
print(unweighted_cm)
print(" ")

# Unweighted confusion matrix, normalized on true labels
unweighted_cm = confusion_matrix(targets, predictions, normalize="true")
print("Unweighted, normalized true labels, CM:")
print(unweighted_cm)
print(" ")

# Unweighted confusion matrix, with raw counts (non-normalized)
unweighted_cm = confusion_matrix(targets, predictions, normalize=None)
print("Unweighted, non-normalized, CM:")
print(unweighted_cm)
print(" ")

# Weighted Confusion Matrix
# Dummy sample weights
sample_weights = np.random.rand(N)

weighted_cm = confusion_matrix(targets, predictions, sample_weights=sample_weights)
print("Weighted CM:")
print(weighted_cm)
