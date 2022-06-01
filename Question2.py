import matplotlib.pyplot as plt # For general plotting
from scipy.stats import multivariate_normal # MVN not univariate

import numpy as np
import math

np.random.seed(7)

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

N = 10000

mu = np.array([[-1, -1], [-1/2, -1/2], [1/2, 1/2], [1, 1]])
Sigma = np.array([
                [[.2, 0],
                [0, .2]],
                [[.3, 0],
                [0, .3]],
                [[.7, 0],
                [0, .7]],
                [[.5, 0],
                [0, .5]]
                ])

priors = np.array([0.2, 0.25, 0.25, 0.3])  # Likelihood of each distribution to be selected   
C = len(priors)

n = mu.shape[1]

X = np.zeros([N, n])
y = np.zeros(N)

# Decide randomly which samples will come from each component
u = np.random.rand(N)
thresholds = np.cumsum(priors)


for c in range(C):
    c_ind = np.argwhere(u <= thresholds[c])[:, 0]  # Get randomly sampled indices for this component
    c_N = len(c_ind)  # No. of samples in this component
    y[c_ind] = c * np.ones(c_N)
    u[c_ind] = 1.1 * np.ones(c_N)  # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
    X[c_ind, :] =  multivariate_normal.rvs(mu[c], Sigma[c], c_N)

# Plot the original data and their true labels
plt.figure(figsize=(12, 10))
plt.plot(X[y==0, 0], X[y==0, 1], 'bo', label="Class 0")
plt.plot(X[y==1, 0], X[y==1, 1], 'rx', label="Class 1");
plt.plot(X[y==2, 0], X[y==2, 1], 'cx', label="Class 2");
plt.plot(X[y==3, 0], X[y==3, 1], 'go', label="Class 3");

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Labels")
plt.tight_layout()
plt.show()

Y = np.array(range(C))
Lambda = np.ones((C, C)) - np.identity(C)
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in Y])
class_priors = np.diag(priors)




class_posteriors = priors.dot(class_cond_likelihoods)

# Class Posterior
# P(yj | x) = p(x | yj) * P(yj) / p(x)
class_posteriors = class_priors.dot(class_cond_likelihoods)


# Conditional RISK - 2 Arrays that hold the risk for each decision
# Sum lamba * P(Y = j | x )
# See Slide 21 of Beyesian Decision Theory
cond_risk = Lambda.dot(class_posteriors)


# Decision Rule
# As of right now, it selects the index that has the lowest value for the specific column
# Value is R(D(x) = i | x)
decisions = np.argmax(class_posteriors, axis=0)


fig = plt.figure(figsize=(12, 10))
marker_shapes = 'ox+*.' # Accomodates up to C=5
marker_colors = 'brgm'

# Get sample class counts
sample_class_counts = np.array([sum(y == j) for j in Y])

# Confusion matrix

conf_mat = np.zeros((C, C))
display_mat = np.zeros((C, C))
for i in Y: # Each decision option
    for j in Y: # Each class label
        ind_ij = np.argwhere((decisions==i) & (y==j))
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j] # Average over class sample count
        display_mat[i,j] =  len(ind_ij)

        # True label = Marker shape; Decision = Marker Color
        marker = marker_shapes[j] + marker_colors[i]
        plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker)

        if i != j:
            plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker, markersize=16)
            


#Confusion Matrix
# TP | FN
# FP |
print("Confusion Matrix (rows: Predicted class, columns: True class):")
print(display_mat)

print("Confusion matrix by average:")
print(conf_mat)

correct_class_samples = np.sum(np.diag(display_mat))
print("Total Mumber of Misclassified Samples: {:.4f}".format(N - correct_class_samples))

prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print("Minimum Probability of Error: ", prob_error)

plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
plt.show()


# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

Nl = np.array([sum(y == l) for l in Y])

# True Negative Probability
ind_00_map = np.argwhere((decisions==0) & (y==0))
p_00_map = len(ind_00_map) / Nl[0]

# False Positive Probability
ind_0x_map = np.argwhere((decisions==0) & (y!=0))
p_0x_map = len(ind_0x_map) / Nl[0]
# False Negative Probability
ind_11_map = np.argwhere((decisions==1) & (y==1))
p_11_map = len(ind_11_map) / Nl[1]
# True Positive Probability
ind_1x_map = np.argwhere((decisions==1) & (y!=1))
p_1x_map = len(ind_1x_map) / Nl[1]

# True Negative Probability
ind_22_map = np.argwhere((decisions==2) & (y==2))
p_22_map = len(ind_22_map) / Nl[2]
# False Positive Probability
ind_2x_map = np.argwhere((decisions==2) & (y!=2))
p_2x_map = len(ind_2x_map) / Nl[2]
# False Negative Probability
ind_33_map = np.argwhere((decisions==3) & (y==3))
p_33_map = len(ind_33_map) / Nl[3]
# True Positive Probability
ind_3x_map = np.argwhere((decisions==3) & (y!=3))
p_3x_map = len(ind_3x_map) / Nl[3]



# Probability of error for MAP classifier, empirically estimated
#prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)
#print(np.array((p_10_map, p_01_map)).shape)
# Display MAP decisions
fig = plt.figure(figsize=(10, 10))

# class 0 circle, class 1 +, correct green, incorrect red
plt.plot(X[ind_00_map, 0], X[ind_00_map, 1], 'og', label="Correct Class 0")
plt.plot(X[ind_0x_map, 0], X[ind_0x_map, 1], 'or', label="Incorrect Class 0")
plt.plot(X[ind_1x_map, 0], X[ind_1x_map, 1], '+r', label="Incorrect Class 1")
plt.plot(X[ind_11_map, 0], X[ind_11_map, 1], '+g', label="Correct Class 1")
plt.plot(X[ind_2x_map, 0], X[ind_2x_map, 1], 'dr', label="Incorrect Class 2")
plt.plot(X[ind_22_map, 0], X[ind_22_map, 1], 'dg', label="Correct Class 2")
plt.plot(X[ind_3x_map, 0], X[ind_3x_map, 1], '^r', label="Incorrect Class 3")
plt.plot(X[ind_33_map, 0], X[ind_33_map, 1], '^g', label="Correct Class 3")

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("MAP Decisions (RED incorrect)")
plt.tight_layout()
plt.show()
