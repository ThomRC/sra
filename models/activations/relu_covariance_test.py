import numpy as np
from scipy.special import erf
from scipy.stats import ortho_group

X_m = np.zeros(784)
X_m[0:150] = 1
X_v = np.diag(np.ones(784)) * 0.1

W1 = ortho_group.rvs(784)[0:512,:]
W2 = ortho_group.rvs(512)
W3 = ortho_group.rvs(512)
W4 = ortho_group.rvs(512)[0:10,:]

mean_s1 = W1 @ X_m
var_s1 = W1 @ X_v @ W1.T

# relu mean
eps = 10 ** -14  # numerical stability constant
alpha = mean_s1 / (np.sqrt(2 * np.diag(var_s1)) + eps)
alpha_erf = erf(alpha)
term1_mean = np.sqrt(0.5 * np.diag(var_s1) / np.pi) * np.exp(-alpha ** 2)
term2_mean = mean_s1 * (1 + alpha_erf) * 0.5
h_m = term1_mean + term2_mean

# relu variance
eps = 10 ** -14  # numerical stability constant
alpha = mean_s1 / (np.sqrt(2 * np.diag(var_s1)) + eps)
alpha_erf = erf(alpha)
h_v = (np.diag(var_s1) / 2) * (1 + alpha_erf) - h_m * (h_m - mean_s1)  # + eps
h_v = np.clip(h_v, a_min=0, a_max = None)

mean_s2 = W2 @ h_m
var_s2 = W2 @ np.diag(h_v) @ W2.T

# relu mean
eps = 10 ** -14  # numerical stability constant
alpha = mean_s2 / (np.sqrt(2 * np.diag(var_s2)) + eps)
alpha_erf = erf(alpha)
term1_mean = np.sqrt(0.5 * np.diag(var_s2) / np.pi) * np.exp(-alpha ** 2)
term2_mean = mean_s2 * (1 + alpha_erf) * 0.5
h_m = term1_mean + term2_mean

# relu variance
eps = 10 ** -14  # numerical stability constant
alpha = mean_s2 / (np.sqrt(2 * np.diag(var_s2)) + eps)
alpha_erf = erf(alpha)
h_v = (np.diag(var_s2) / 2) * (1 + alpha_erf) - h_m * (h_m - mean_s2)  # + eps
h_v = np.clip(h_v, a_min=0, a_max = None)

mean_s3 = W3 @ h_m
var_s3 = W3 @ np.diag(h_v) @ W3.T

# relu mean
eps = 10 ** -14  # numerical stability constant
alpha = mean_s3 / (np.sqrt(2 * np.diag(var_s3)) + eps)
alpha_erf = erf(alpha)
term1_mean = np.sqrt(0.5 * np.diag(var_s3) / np.pi) * np.exp(-alpha ** 2)
term2_mean = mean_s3 * (1 + alpha_erf) * 0.5
h_m = term1_mean + term2_mean

# relu variance
eps = 10 ** -14  # numerical stability constant
alpha = mean_s3 / (np.sqrt(2 * np.diag(var_s3)) + eps)
alpha_erf = erf(alpha)
h_v = (np.diag(var_s3) / 2) * (1 + alpha_erf) - h_m * (h_m - mean_s3)  # + eps
h_v = np.clip(h_v, a_min=0, a_max = None)

mean_s4 = W4 @ h_m
var_s4 = W4 @ np.diag(h_v) @ W4.T

# relu mean
eps = 10 ** -14  # numerical stability constant
alpha = mean_s4 / (np.sqrt(2 * np.diag(var_s4)) + eps)
alpha_erf = erf(alpha)
term1_mean = np.sqrt(0.5 * np.diag(var_s4) / np.pi) * np.exp(-alpha ** 2)
term2_mean = mean_s4 * (1 + alpha_erf) * 0.5
h_m = term1_mean + term2_mean

# relu variance
eps = 10 ** -14  # numerical stability constant
alpha = mean_s4 / (np.sqrt(2 * np.diag(var_s4)) + eps)
alpha_erf = erf(alpha)
h_v = (np.diag(var_s4) / 2) * (1 + alpha_erf) - h_m * (h_m - mean_s4)  # + eps
h_v = np.clip(h_v, a_min=0, a_max = None)
