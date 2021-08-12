# ====================================================================================== #
# For keeping track of maxent models.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from coniii.samplers import Metropolis, Potts3
import coniii.utils as cutils
import numpy as np
from itertools import combinations
from scipy.special import binom


class LargeIsing():
    """Class to store means, pairwise, triplet, and 4th order correlations using
    Metropolis sampling.
    """
    def __init__(self, multipliers,
                 sample_size=10_000_000,
                 n_iters=1_000,
                 burn_in=10_000,
                 iprint=True):
        """
        Parameters
        ----------
        multipliers : list of ndarray
        sample_size : int, 10_000_000
        n_iters : int, 1_000
        burn_in : int, 10_000
        """
        
        self.n = len(multipliers[0])
        self.sample_size = sample_size
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.iprint = iprint

        self.update_multipliers(multipliers)

    def update_multipliers(self, multipliers):
        """Update multipliers, resample from the defined distribution, and re-calculate
        observables.

        Correlations are stored in list self.corr in order of order.

        Parameters
        ----------
        multipliers : ndarray
        """

        calc_e, calc_observables = cutils.define_ising_helper_functions()[:2]
        self.multipliers = np.concatenate(multipliers)

        if self.iprint: print("Generating samples...")
        self.sampler = Metropolis(self.n, self.multipliers, calc_e)
        self.sampler.generate_samples_parallel(self.sample_size,
                                               self.n_iters,
                                               burn_in=self.burn_in)
        samples = self.sampler.samples
        if self.iprint: print("Done generating samples.")
        
        # correlations
        if self.iprint: print("Calculating correlations...")
        # means and pairwise corr
        corr = calc_observables(samples).mean(0)
        corr = [corr[:self.n], corr[self.n:]]
        
        # third order corr
        #c = np.zeros(int(binom(self.n,3)))
        #counter = 0
        #for ijk in combinations(range(self.n), 3):
        #    c[counter] = np.prod(samples[:,ijk],1).mean()
        #    counter += 1
        #corr.append(c)

        # fourth order corr
        #c = np.zeros(int(binom(self.n,4)))
        #counter = 0
        #for ijk in combinations(range(self.n), 4):
        #    c[counter] = np.prod(samples[:,ijk],1).mean()
        #    counter += 1
        #corr.append(c)
        if self.iprint: print("Done calculating correlations.")

        # probability distribution over sample
        if self.iprint: print("Calculating p(s)...")
        states, p = np.unique(samples, axis=0, return_counts=True)
        p = p / p.sum()  
        if self.iprint: print("Done calculating p(s).")
        
        self.corr = corr
        self.states = states
        self.p = p
#end LargeIsing


class LargePotts3():
    """Class to store means, pairwise, triplet, and 4th order correlations using
    Metropolis sampling.
    """
    def __init__(self, multipliers,
                 sample_size=10_000_000,
                 n_iters=1_000,
                 burn_in=10_000,
                 rng=None,
                 iprint=True):
        """
        Parameters
        ----------
        multipliers : list of ndarray
        sample_size : int, 10_000_000
        n_iters : int, 1_000
        burn_in : int, 10_000
        """
        
        self.n = int(len(multipliers[0])//3)
        self.sample_size = sample_size
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.rng = rng or np.random
        self.iprint = iprint
        self.calc_e, self.calc_observables = cutils.define_potts_helper_functions(3)[:2]

        self.update_multipliers(multipliers)

    def update_multipliers(self, multipliers):
        """Update multipliers, resample from the defined distribution, and re-calculate
        observables.

        Correlations are stored in list self.corr in order of order.

        Parameters
        ----------
        multipliers : ndarray
        """
        
        n = self.n
        self.multipliers = np.concatenate(multipliers)

        if self.iprint: print("Generating samples...")
        self.sampler = Potts3(n, self.multipliers, self.calc_e, rng=self.rng)
        self.sampler.generate_samples_parallel(self.sample_size,
                                               self.n_iters,
                                               burn_in=self.burn_in)
        samples = self.sampler.samples
        if self.iprint: print("Done generating samples.")
        
        # correlations
        if self.iprint: print("Calculating correlations...")
        # means and pairwise corr
        corr = [np.zeros(n*3), np.zeros(n*(n-1)//2)]
        counter = 0
        for i in range(3):
            for j in range(n):
                corr[0][counter] = (samples[:,j]==i).mean()
                counter += 1

        counter = 0
        for i in range(n-1):
            for j in range(i+1,n):
                corr[1][counter] = (samples[:,i]==samples[:,j]).mean()
                counter += 1
        #corr = calc_observables(samples).mean(0)
        #corr = [corr[:self.n], corr[self.n:]]
        
        # third order corr
        #c = np.zeros(int(binom(self.n,3)))
        #counter = 0
        #for ijk in combinations(range(self.n), 3):
        #    c[counter] = np.prod(samples[:,ijk],1).mean()
        #    counter += 1
        #corr.append(c)

        # fourth order corr
        #c = np.zeros(int(binom(self.n,4)))
        #counter = 0
        #for ijk in combinations(range(self.n), 4):
        #    c[counter] = np.prod(samples[:,ijk],1).mean()
        #    counter += 1
        #corr.append(c)
        if self.iprint: print("Done calculating correlations.")

        # probability distribution over sample
        if self.iprint: print("Calculating p(s)...")
        states, p = np.unique(samples, axis=0, return_counts=True)
        p = p / p.sum()  
        if self.iprint: print("Done calculating p(s).")
        
        self.corr = corr
        self.states = states
        self.p = p
#end LargePotts3
