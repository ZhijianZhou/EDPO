import numpy as np


def kl_divergence_normal(mu_P, sigma_P, mu_Q):
    """
    Compute the Kullback-Leibler (KL) divergence between two normal distributions using PyTorch.
    
    Args:
    - mu_P (float or tensor): Mean of the first normal distribution (P)
    - sigma_P (float or tensor): Standard deviation of the first normal distribution (P)
    - mu_Q (float or tensor): Mean of the second normal distribution (Q)
    - sigma_Q (float or tensor): Standard deviation of the second normal distribution (Q)
    
    Returns:
    - float or tensor: KL divergence from P to Q
    """
    # First term: log(sigma_Q / sigma_P)
    
    # Second term: (sigma_P^2 + (mu_P - mu_Q)^2) / (2 * sigma_Q^2)
    term2 = (sigma_P**2 + (mu_P - mu_Q)**2) / (2 * sigma_P**2+1e-8)
    
    # Third term: -1/2
    term3 = -0.5
    
    # Calculate the KL divergence
    kl_div = term2 + term3
    return kl_div

def rmsd(A, B=None):
    """
    Calculate the RMSD (Root Mean Square Deviation) between two 2D matrices A and B.
    If B is not provided, it defaults to a zero matrix.

    Parameters:
    A: numpy.ndarray, shape (m, n)
    B: numpy.ndarray, shape (m, n), default is None. If None, B will be set to a zero matrix.
    
    Returns:
    float: RMSD value
    """
    # If B is None, set B to a zero matrix with the same shape as A
    if B is None:
        B = np.zeros_like(A)

    # Ensure matrices A and B have the same shape
    if A.shape != B.shape:
        raise ValueError("The input matrices A and B must have the same shape")
    
    # Calculate the squared differences between matrices A and B
    diff = A - B
    squared_diff = np.square(diff)
    
    # Compute the root mean square deviation (RMSD)
    rmsd_value = np.sqrt(np.mean(squared_diff))
    
    return rmsd_value