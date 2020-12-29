# ====================================================================================== #
# Module for solving maxent problem on C elegans data set.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import multinomial



def _indpt(X):
    """Solve independent spin model.

    Parameters
    ----------
    X : ndarray
        Dimension (n_samples, n_neurons).
    
    Returns
    -------
    ndarray
        Solved field returned as fields for 0 state for all spins, then 1 state for all
        spins, 2 state for all spins to form a total length of 3 * N.
    """
    
    p = np.vstack([(X==i).mean(0) for i in range(3)])
    p = p.T

    h = np.zeros((p.shape[0], 3))

    # solve each spin
    def p_ind(h):
        return np.exp(h - logsumexp(h))

    for i in range(p.shape[0]):
        pi = p[i]
        def cost(h):
            return np.linalg.norm(p_ind(h) - pi)

        h[i] = minimize(cost, [0,0,0])['x']

    # set the third field to zero (this is our normalized representation)
    h -= h[:,2][:,None]

    return h.T.ravel()

def indpt(X, s=1):
    """Solve independent spin model with Gaussian prior.

    Parameters
    ----------
    X : ndarray
        Dimension (n_samples, n_neurons).
    s : float, 1
        Std of Gaussian prior on fields.
    
    Returns
    -------
    ndarray
        Solved field returned as fields for 0 state for all spins, then 1 state for all
        spins, 2 state for all spins to form a total length of 3 * N.
    """
    
    p = np.vstack([(X==i).mean(0) for i in range(3)])
    p = p.T

    # breakdown in number of occurrences per state
    n = np.vstack([np.bincount(X[:,i], minlength=3) for i in range(X.shape[1])])

    h = np.zeros((p.shape[0], 3))

    # solve each spin
    def p_ind(h):
        return np.exp(h - logsumexp(h))

    for i in range(p.shape[0]):
        pi = p[i]

        def cost(h):
            logpdf = multinomial.logpmf(n[i], n=len(X), p=p_ind(h))
            return logpdf + (h**2).sum() / s**2

        h[i] = minimize(cost, [0,0,0])['x']

    # set the third field to zero (this is our normalized representation)
    h -= h[:,2][:,None]

    return h.T.ravel()
