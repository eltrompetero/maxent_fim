# ====================================================================================== #
# Module for projecting FIM tangent space onto dissenting blocs.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from .utils import *
from coniii.enumerate import fast_logsumexp
import os
calc_e, calc_observables, _ = define_ising_helper_functions()


class BlocProjection():
    def __init__(self, tanSpace, eigvec, names, eigix=0, eps=1e-5):
        """
        Parameters
        ----------
        tanSpace : IsingFisherCurvatureMethod2 
            Tangent space.
        eigvec : ndarray
        names : list of str
        eigix : int, 0
            Which direction in tangent space to take corresponding to eigenmatrices of
            FIM.
        eps : float, 1e-5
            For calculating finite derivative.
        """

        assert 0<eps<.1

        self.tanSpace = tanSpace
        self.eigvec = eigvec
        self.eps = eps
        self.names = names
        self.n = tanSpace.n

        dE = calc_e(tanSpace.allStates.astype(np.int64),
                    (eigvec[:,eigix][:,None]*tanSpace.dJ).sum(0))*eps
        E = np.log(tanSpace.p)
        self.pplus = np.exp(E+dE - fast_logsumexp(E+dE)[0])  # modified probability distribution
        self.pminus = np.exp(E-dE - fast_logsumexp(E-dE)[0])  # modified probability distribution

        self.pkplusdE = np.zeros(tanSpace.n//2+1)
        self.pkminusdE = np.zeros(tanSpace.n//2+1)
        for k in range(5):
            self.pkplusdE[k] = self.pplus[np.abs(tanSpace.allStates.sum(1))==(self.n-k*2)].sum()
            self.pkminusdE[k] = self.pminus[np.abs(tanSpace.allStates.sum(1))==(self.n-k*2)].sum()

    def dlogp(self, k, run_checks=False):
        """Calculate change in log likelihood (or change in energy) when probability
        distribution is perturbed infinitesmically (linearized regime) in a direction in the
        tangent space.
        
        Parameters
        ----------
        k : int
            Number of votes in the dissenting coalition.
        run_checks : bool, False
        
        Returns
        -------
        ndarray
            Log base 10 probability of bloc.
        ndarray
            Change in log base 10 probability of bloc.
        list of str
            Sets of components that belong to each bloc.
        """
        
        assert (self.tanSpace.n//2)>=k>=1
        
        # use symmetry to extract dissenting coalitions including flip symmetry
        projVotes = np.vstack([i for i in self.tanSpace.allStates
                               if abs(i.sum())==(self.n-k*2)]).astype(np.int64)
        projVotes = [projVotes[[i,len(projVotes)-1-i],:] for i in range(len(projVotes)//2)]
        namesInBlocs = [', '.join(np.sort(self.names[pv[0]==-np.sign(pv[0].sum())]))
                       for pv in projVotes]
        if run_checks:
            assert all([(pv[0]==-pv[1]).all() for pv in projVotes])
            assert all([len(i)==(2*k+k-1+k-1) for i in namesInBlocs])

        # get log likelihood ratios when the energy is changed along the chosen direction
        logp = np.zeros(len(projVotes))
        dlogp = np.zeros(len(projVotes))
        for i,pv in enumerate(projVotes):
            thisp = 0
            thispplusdE = 0
            thispminusdE = 0
            for pv_ in pv:
                ix = (self.tanSpace.allStates==pv_[None,:]).all(1)
                thisp += self.tanSpace.p[ix]
                thispplusdE += self.pplus[ix]
                thispminusdE += self.pminus[ix]
            logp[i] = np.log10(thisp)
            dlogp[i] = (np.log10(thispplusdE) - np.log10(thispminusdE)) / (2*self.eps)

        return logp, dlogp, namesInBlocs
