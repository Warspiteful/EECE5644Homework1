"""Utilities for manipulating or generating data."""

import numpy as np


def data_augmentation(X, n):
    X_aug = np.ones(shape=(len(X), 1))

    for order in range(1, n + 1):
        X_aug = np.column_stack((X_aug, X ** order))

    return X_aug


# Generating synthetic data
def synthetic_data(theta, b, N):
    # Generate y = Xw + b + noise
    X = np.random.normal(0, 1, (N, len(theta)))
    y = X.dot(theta) + b + np.random.normal(0, 0.01, N)
    return X, y


# Breaks the matrix X and vector y into batches
def batchify(X, y, batch_size):
    N = X.shape[0]
    X_batch = []
    y_batch = []
    # Iterate over N in batch_size steps, last batch may be < batch_size
    for i in range(0, N, batch_size):
        nxt = min(i + batch_size, N + 1)
        X_batch.append(X[i:nxt, :])
        y_batch.append(y[i:nxt])

    return X_batch, y_batch