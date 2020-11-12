# ====================================================================================== #
# Module for solving maxent problem on C elegans data set.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from scipy.optimize import minimize
from scipy.special import logsumexp



def indpt(X):
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
    h -= h[:,2][:,None]

    return h.T.ravel()

