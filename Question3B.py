import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

import numpy as np
from scipy.stats import multivariate_normal # MVN not univariate

import pandas

from modules import models, prob_utils

from collections import defaultdict

data = defaultdict(list) # each value in each column is appended to a list

with open('./Homework1/human_data.txt') as f:
    lines = f.readlines()

with open('./Homework1/human_labels.txt') as f:
    labels = f.readlines()
data = []
for index in range(len(lines)):
    data_imm = lines[index].split()
    test = np.array([float(num) for num in data_imm])
    data.append(test)
lines = np.array(data)

data = []
for index in range(len(labels)):
    data.append(int(labels[index].split()[0]))
labels = np.array(data)
np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)

label_set = set(labels) 
indexes = []
for label in label_set:
    indexes.append([i for i,val in enumerate(labels) if val==label])

datasets = []
for label in range(len(label_set)):
    datasets.append([lines[indexer] for indexer in indexes[label]])
aa = sum([lines[row_index][0] for row_index in indexes[0]])

mu = [] 
for label in range(len(label_set)):
    index_means = []
    for ind in range(lines.shape[1]):
        index_means.append(np.mean( [lines[row_index][ind] for row_index in indexes[label]]))
    
    mu.append(np.array(index_means))
mu = np.array(mu)
Sigma = []
n = mu.shape[1] 

offset = np.identity(n)
for label in range(len(label_set)):
    
    Sigma.append(np.cov([lines[row_index][ind] for row_index in indexes[label]]) +  .5*np.identity(n))
    

Sigma = np.array(Sigma)

counts = [len(indexes[i]) for i in range(len(indexes))]
size = len(lines)


priors = np.array([counts[index]/size for index in range(len(label_set))])  
C = len(priors)
# Likelihood of each distribution to be selected   
class_priors = np.diag(priors)





X = lines
Y = np.array(list(label_set))
y = labels

# Decide randomly which samples will come from each component

# Transpose XT into shape [N, n] to fit into algorithm


Lambda = np.ones((C, C)) - np.identity(C)
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in range(len(Y))])

class_posteriors = priors.dot(class_cond_likelihoods)

# Class Posterior
# P(yj | x) = p(x | yj) * P(yj) / p(x)
class_posteriors = class_priors.dot(class_cond_likelihoods)


# Decision Rule
# As of right now, it selects the index that has the lowest value for the specific column
# Value is R(D(x) = i | x)
decisions = np.argmax(class_posteriors, axis=0)+1


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
