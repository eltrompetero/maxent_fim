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


class Independent3():
    def __init__(self, X=None):
        if not X is None:
            self.set_data(X)

    def set_data(self, X):
        """
        Parameters
        ----------
        X : ndarray
            Dimension (n_samples, n_neurons).
        """

        self.X = X
        self.p = np.vstack([(X==i).mean(0) for i in range(3)])
        self.p = self.p.T

    def solve(self, s=1, full_output=False):
        """Solve independent spin model with Gaussian prior.

        Parameters
        ----------
        s : float, 1
            Std of Gaussian prior on fields.
        full_output : bool, False
        
        Returns
        -------
        ndarray
            Solved field returned as fields for 0 state for all spins, then 1 state for
            all spins, 2 state for all spins to form a total length of 3 * N.
        list of dict (optional)
        """
        
        # breakdown in number of occurrences per state
        n = np.vstack([np.bincount(self.X[:,i], minlength=3) for i in range(self.X.shape[1])])

        h = np.zeros((self.p.shape[0], 3))

        # solve each spin separately
        soln = []
        for i in range(self.p.shape[0]):
            soln.append( minimize(lambda h: self.cost(h, n[i], s), [0,0,0]) )
            h[i] = soln[-1]['x']

        # set the third field to zero (this is our normalized representation)
        h -= h[:,2][:,None]
    
        if full_output:
            return h.T.ravel(), soln
        return h.T.ravel()

    def p_ind(self, h):
        """Set of probabilities for a single spin.

        Parameters
        ----------
        h : ndarray

        Returns
        -------
        ndarray
            The three probabilities.
        """

        return np.exp(h - logsumexp(h))

    def cost(self, h, n, s):
        """Log likelihood cost for a single spin.

        Parameters
        ----------
        h : ndarray
            Fields.
        n : ndarray
            Number of observations per state.
        s : float
            Std of Gaussian prior.

        Returns
        -------
        ndarray
        """

        logpdf = multinomial.logpmf(n, n=n.sum(), p=self.p_ind(h))
        return -logpdf / n.sum() + (h**2).sum() / 2 / s**2

    def cost_with_s(self, s_range):
        """Optimal cost as a function of the prior width. There will be a separate set of
        values for each spin.
        
        Parameters
        ----------
        s_range : ndarray

        Returns
        -------
        ndarray
            (n_spins, s_range.size)
        """
        
        c = np.zeros((self.X.shape[1], s_range.size))

        for i, s in enumerate(s_range):
            soln = self.solve(s=s, full_output=True)[1]
            c[:,i] = [s['fun'] for s in soln]

        return c
#end Independent3

