import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

import numpy as np
from scipy.stats import multivariate_normal # MVN not univariate

import pandas

from modules import models, prob_utils

from collections import defaultdict

data = defaultdict(list) # each value in each column is appended to a list

windata = pandas.read_csv('./Homework1/WhiteWine.csv', sep = ';', index_col="quality")  


np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title


size = len(windata)

labels = set()
for ax in windata.index:
    labels.add(ax)


# for index in labels:
  #  print("Label: {:.4f}".format(index))
  #  print(np.mean(fixed_acidity[index].to_numpy()))

counts = np.array([len(windata["fixed acidity"][index].to_numpy()) for index in labels])




mu = []
for index in labels:
    mu.append([np.mean(windata[feature][index]) for feature in windata])
mu = np.array(mu)
n = mu.shape[1]
Sigma = []

offset = np.identity(n)
for index in labels:
    Sigma.append(np.cov([windata[feature][index].to_numpy() for feature in windata]) +  5*np.identity(n))

priors = np.array([counts[index]/size for index in range(len(labels))])  
C = len(priors)
# Likelihood of each distribution to be selected   
class_priors = np.diag(priors)





X = windata.values
Y = np.array(list(labels))
y = np.array(windata.index)

# Decide randomly which samples will come from each component

# Transpose XT into shape [N, n] to fit into algorithm


class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in range(len(Y))])


# Class Posterior
# P(yj | x) = p(x | yj) * P(yj) / p(x)
class_posteriors = class_priors.dot(class_cond_likelihoods)


# Decision Rule
decisions = np.argmax(class_posteriors, axis=0)+3


# Get sample class counts
sample_class_counts = np.array([sum(y == j) for j in Y])

# Confusion matrix

conf_mat = np.zeros((C, C))
display_mat = np.zeros((C,C))
for i in range(len(Y)): # Each decision option
    for j in range(len(Y)): # Each class label
        ind_ij = np.argwhere((decisions==Y[i]) & (y==Y[j]))
        display_mat[i, j] = len(ind_ij) # Average over class sample count
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j]



print("Mean Vectors:")
print(mu)
print("Covariance Matrix:")
print(Sigma)
print("Priors", priors)

#Confusion Matrix
# TP | FN
# FP |
print("Confusion matrix:")
print(display_mat)


misclass = size - sum(np.diag(display_mat))
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / size)

print("Minimum Probability of Error: ", prob_error)
print("Misclassified Samples: ", misclass)



# Perform PCA on transposed GMM variable X
_, _, Z = models.perform_pca(X)

# Add back mean vector to PC projections if you want PCA reconstructions
Z_GMM = Z + np.mean(X, axis=0)

# Plot original data vs PCA reconstruction data
fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(211, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("x3")
ax1.set_title("x ~ {}D GMM data".format(n))

ax2 = fig.add_subplot(212, projection='3d')
ax2.scatter(Z_GMM[:, 0], Z_GMM[:, 1], Z_GMM[:, 2])
ax2.set_xlabel("z1")
ax2.set_ylabel("z2")
ax2.set_zlabel("z3")
ax2.set_title("PCA projections of {}D GMM data".format(n))
plt.show()

# Let's see what it looks like only along the first two PCs
fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(211)
ax1.scatter(X[:, 0], X[:, 1])
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("x ~ {}D GMM data to 2D space".format(n))

ax2 = fig.add_subplot(212)
ax2.scatter(Z_GMM[:, 0], Z_GMM[:, 1])
ax2.set_xlabel("z1")
ax2.set_ylabel("z2")
ax2.set_title("PCA projections of {}D GMM data to 2D space".format(n))
plt.show()
