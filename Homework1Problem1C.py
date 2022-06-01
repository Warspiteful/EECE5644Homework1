# Mean and covariance of data pdfs conditioned on labels
import matplotlib.pyplot as plt # For general plotting

import numpy as np

import math


from scipy.stats import multivariate_normal # MVN not univariate

np.set_printoptions(suppress=True)

N = 10000
mu = np.array([[-1/2, -1/2, -1/2], [1, 1, 1]])
Sigma = np.array([[[1, -0.5, 0.3],
                   [-0.5, 1, -0.5],
                   [0.3, -0.5, 1]],
                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                  [-0.2, 0.3, 1]]])

# Determine dimensionality from mixture PDF parameters
n = mu.shape[1]

print(Sigma.shape)
# Class priors
priors = np.array([0.65, 0.35])  
C = len(priors)
# Decide randomly which samples will come from each component (taking class 1 from standard normal values above 0.35)
labels = np.random.rand(N) >= priors[0]
L = np.array(range(C))
Nl = np.array([sum(labels == l) for l in L])


# Draw samples from each class pdf
X = np.zeros((N, n))
X[labels == 0, :] =  multivariate_normal.rvs(mu[0], Sigma[0], Nl[0])
X[labels == 1, :] =  multivariate_normal.rvs(mu[1], Sigma[1], Nl[1])
mu_estimate = np.array([[(sum(X[labels == 0,0]) / Nl[0]), (sum(X[labels == 0,1]) / Nl[0]), (sum(X[labels == 0,2]) / Nl[0])], 
                        [(sum(X[labels == 1,0]) / Nl[1]), (sum(X[labels == 1,1]) / Nl[1]), (sum(X[labels == 1,2]) / Nl[1])]                                        ])

print("Means:")

print(mu_estimate)

sigma_estimate = np.array(
    [[
                    [
                    sum(pow(X[labels == 0,0] - mu[0][0],2)) / (Nl[0]-1), sum((X[labels == 0,0] - mu[0][0])*(X[labels == 0,1] - mu[0][1])) / (Nl[0]-1), sum((X[labels == 0,0] - mu[0][0])*(X[labels == 0,2] - mu[0][2])) / (Nl[0]-1)],
                    [sum((X[labels == 0,1] - mu[0][1])*(X[labels == 0,0] - mu[0][0])) / (Nl[0]-1), sum(pow(X[labels == 0,1] - mu[0][1],2)) / (Nl[0]-1), sum((X[labels == 0,1] - mu[0][1])*(X[labels == 0,2] - mu[0][2])) / (Nl[0]-1)],
                    [sum((X[labels == 0,2] - mu[0][2])*(X[labels == 0,0] - mu[0][0])) / (Nl[0]-1), sum((X[labels == 0,2] - mu[0][2])*(X[labels == 0,1] - mu[0][1])) / (Nl[0]-1), sum(pow(X[labels == 0,2] - mu[0][2], 2)) / (Nl[0]-1)]
                    ],
                    [
                    [sum(pow(X[labels == 1,0] - mu[1][0],2)) / (Nl[1]-1), sum((X[labels == 1,0] - mu[1][0])*(X[labels == 1,1] - mu[1][1])) / (Nl[1]-1), sum((X[labels == 1,0] - mu[1][0])*(X[labels == 1,2] - mu[1][2])) / (Nl[1]-1)],
                    [sum((X[labels == 1,1] - mu[1][1])*(X[labels == 1,0] - mu[1][0])) / (Nl[1]-1), sum(pow(X[labels == 1,1] - mu[1][1],2)) / (Nl[1]-1), sum((X[labels == 1,1] - mu[1][1])*(X[labels == 1,2] - mu[1][2])) / (Nl[1]-1)],
                    [sum((X[labels == 1,2] - mu[1][2])*(X[labels == 1,0] - mu[1][0])) / (Nl[1]-1), sum((X[labels == 1,2] - mu[1][2])*(X[labels == 1,1] - mu[1][1])) / (Nl[1]-1), sum(pow(X[labels == 1,2] - mu[1][2], 2)) / (Nl[1]-1)]
                    ]
                    ]
                )

print("Sigma:")
print(sigma_estimate)

# MAP classifier (is a special case of ERM corresponding to 0-1 loss)
# 0-1 loss values yield MAP decision rule
Lambda = np.ones((C, C)) - np.identity(C)


class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

# Gamma threshold for MAP decision rule (remove Lambdas and you obtain same gamma on priors only; 0-1 loss simplification)
gamma_map = (Lambda[1,0] - Lambda[0,0]) / (Lambda[0,1] - Lambda[1,1]) * priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]

# Plot the original data and their true labels
fig = plt.figure(figsize=(10, 10))
plt.plot(X[labels==0, 0], X[labels==0, 1], 'bo', label="Class 0")
plt.plot(X[labels==1, 0], X[labels==1, 1], 'k+', label="Class 1")

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Class Labels")
plt.tight_layout()

from sys import float_info # Threshold smallest positive floating value

# Generate ROC curve samples
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    sorted_score = sorted(discriminant_score)

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] + 
             sorted_score +
             [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))

    return roc, taus


Lambda = np.ones((C, C)) - np.identity(C)

class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

# Gamma threshold for MAP decision rule (remove Lambdas and  you obtain same gamma on priors only; 0-1 loss simplification)
gamma_map = priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]

decisions_map = discriminant_score_erm >= np.log(gamma_map)

# True Negative Probability
ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
p_00_map = len(ind_00_map) / Nl[0]
# False Positive Probability
ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
p_10_map = len(ind_10_map) / Nl[0]
# False Negative Probability
ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
p_01_map = len(ind_01_map) / Nl[1]
# True Positive Probability
ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
p_11_map = len(ind_11_map) / Nl[1]


# Construct the ROC for ERM by changing log(gamma)
roc_erm, _ = estimate_roc(discriminant_score_erm, labels)
roc_map = np.array((p_10_map, p_11_map))

plt.ioff() # These are Jupyter only lines to avoid showing the figure when I don't want
fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
plt.ion() # Re-activate "interactive" mode

ax_roc.plot(roc_erm[0], roc_erm[1])
ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Minimum P(Error) MAP", markersize=16)
ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.grid(True)

def perform_lda(X, mu, Sigma, C=2):
    """  Fisher's Linear Discriminant Analysis (LDA) on data from two classes (C=2).

    In practice the mean and covariance parameters would be estimated from training samples.
    
    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.
        mu: Mean vector [C, n].
        Sigma: Covariance matrices [C, n, n].

    Returns:
        w: Fisher's LDA project vector, shape [n, 1].
        z: Scalar LDA projections of input samples, shape [N, 1].
    """
    
    mu = np.array([mu[i].reshape(-1, 1) for i in range(C)])

    cov = np.array([Sigma[i].T for i in range(C)])

    # Determine between class and within class scatter matrix
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]

    # Regular eigenvector problem for matrix Sw^-1 Sb
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]

    # Extract corresponding sorted eigenvectors
    U = U[:, idx]

    # First eigenvector is now associated with the maximum eigenvalue, mean it is our LDA solution weight vector
    w = U[:, 0]

    # Scalar LDA projections in matrix form
    z = X.dot(w)

    return w, z





# Fisher LDA Classifer (using true model parameters)
weight, discriminant_score_lda = perform_lda(X, mu_estimate, sigma_estimate)

# Estimate the ROC curve for this LDA classifier
roc_lda, tau_lda = estimate_roc(discriminant_score_lda, labels)

# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
prob_error_lda = np.array((roc_lda[0,:], 1 - roc_lda[1,:])).T.dot(Nl.T / N)

# Min prob error
min_prob_error_lda = np.min(prob_error_lda)
min_ind = np.argmin(prob_error_lda)


print("LDA Weight Vector: ", weight)

print("Index of Minumum Probability Error: ", min_ind)

print("Empirical Estimated Probability of Error: {:.4f}".format(min_prob_error_lda))
print("Empirical Threshold: {:.4f}".format(math.exp(float(tau_lda[min_ind]))))





ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
# Display the estimated ROC curve for LDA and indicate the operating points
# with smallest empirical error probability estimates (could be multiple)
ax_roc.plot(roc_lda[0], roc_lda[1], 'b:')
ax_roc.plot(roc_lda[0, min_ind], roc_lda[1, min_ind], 'r.', label="Minimum P(Error) LDA", markersize=16)
ax_roc.set_title("ROC Curves for ERM and LDA")
ax_roc.legend()
plt.grid(True)

# Use min-error threshold
decisions_lda = discriminant_score_lda >= tau_lda[min_ind]

# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_lda = np.argwhere((decisions_lda==0) & (labels==0))
p_00_lda = len(ind_00_lda) / Nl[0]
# False Positive Probability
ind_10_lda = np.argwhere((decisions_lda==1) & (labels==0))
p_10_lda = len(ind_10_lda) / Nl[0]
# False Negative Probability
ind_01_lda = np.argwhere((decisions_lda==0) & (labels==1))
p_01_lda = len(ind_01_lda) / Nl[1]
# True Positive Probability
ind_11_lda = np.argwhere((decisions_lda==1) & (labels==1))
p_11_lda = len(ind_11_lda) / Nl[1]


# Display LDA decisions
fig = plt.figure(figsize=(10, 10))

# class 0 circle, class 1 +, correct green, incorrect red
plt.plot(X[ind_00_lda, 0], X[ind_00_lda, 1], 'og', label="Correct Class 0")
plt.plot(X[ind_10_lda, 0], X[ind_10_lda, 1], 'or', label="Incorrect Class 0")
plt.plot(X[ind_01_lda, 0], X[ind_01_lda, 1], '+r', label="Incorrect Class 1")
plt.plot(X[ind_11_lda, 0], X[ind_11_lda, 1], '+g', label="Correct Class 1")

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("LDA Decisions (RED incorrect)")
plt.tight_layout()

print("Smallest P(error) for LDA = {}".format(min_prob_error_lda))
plt.show(block=True)