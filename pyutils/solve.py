# ====================================================================================== #
# Module for solving maxent problem on C elegans data set.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import multinomial
from scipy.interpolate import interp1d



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
    def __init__(self, X=None, alpha=2.):
        """
        Parameters
        ----------
        X : ndarray
            Data.
        alpha : float, 2.
            Cost exponent.
        """

        if not X is None:
            self.set_data(X)
        self.alpha = alpha

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
        return -logpdf / n.sum() + (np.abs(h)**self.alpha).sum() / 2 / s**self.alpha

    def cost_with_s(self, s_range, n_cpus=None):
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
        
        if n_cpus==1:
            c = np.zeros((self.X.shape[1], s_range.size))

            for i, s in enumerate(s_range):
                soln = self.solve(s=s, full_output=True)[1]
                c[:,i] = [s['fun'] for s in soln]
        else:
            def loop_wrapper(s):
                soln = self.solve(s=s, full_output=True)[1]
                return [s['fun'] for s in soln]
            
            with threadpool_limits(limits=1, user_api='blas'):
                with Pool() as pool:
                    c = np.vstack(list(pool.map(loop_wrapper, s_range))).T
        return c

    def optimize_s(self, prior_range=None, n_interp=32):
        """Find midpoint of hyperparameter s, the width of the prior.

        Parameters
        ----------
        prior_range : ndarray, None
            By default set to 10^-1 and 10^-2.5
        n_interp : int, 32

        Returns
        -------
        float
        """
        
        if prior_range is None:
            prior_range = np.logspace(-1, 2.5, n_interp)

        logl = self.cost_with_s(prior_range).mean(0)

        # interpolate
        spline = interp1d(np.log(prior_range), logl, kind='cubic', bounds_error=False)

        # get the middle point
        midval = (logl[0] + logl[-1])/2
        smid = np.exp(minimize(lambda x: (spline(x) - midval)**2,
                               np.log(prior_range[prior_range.size//2]),
                               bounds=[np.log(prior_range[[0,-1]])])['x'])
        
        return smid
#end Independent3

