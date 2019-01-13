# Useful functions for setting up maxent problem on Super Court.
# Author: Eddie Lee, edl56@cornell.edu
import numpy as np
from numba import njit,jit
from coniii.utils import pair_corr


def define_helpers(N, nonVotingPairsIx):
    """Define energy and observable calculation with helper functions.

    Parameters
    ----------
    N : int
    nonVotingPairsIx : list
        Indices of pairs to avoid in order that would be given by itertools.combinations.

    Returns
    -------
    function
        calc_e
    function
        calc_observables
    function
        mch_approximation
    """

    @njit
    def inside(x, val):
        """Check if val is in vector x."""
        for i in x:
            if i==val:
                return True
        return False

    @njit
    def fast_sum(J, s, J0ix):
        e = np.zeros((s.shape[0]))
        for n in range(s.shape[0]):
            k = 0
            ixCounter = 0
            for i in range(s.shape[1]-1):
                for j in range(i+1,s.shape[1]):
                    if not inside(J0ix,k):
                        e[n] += J[ixCounter]*s[n,i]*s[n,j]
                        ixCounter += 1
                    k += 1
        return e

    def calc_e_subsystem(s, params, nonVotingPairsIx=nonVotingPairsIx):
        """
        Parameters
        ----------
        s : 2D ndarray
            state either {0,1} or {+/-1}.
        params : ndarray
            (h,J) vector.
        """

        e = -fast_sum(params, s, nonVotingPairsIx)
        return e

    @njit
    def calc_observables(sample):
        """Only calculate for pairs that have voted together at all."""
        obs = np.zeros((len(sample), N*(N-1)//2-len(nonVotingPairsIx)))
        obsix = 0
        counter = 0
        nonVotingPairsIxCounter = 0
        for i in range(N-1):
            for j in range(i+1,N):
                if counter!=nonVotingPairsIx[nonVotingPairsIxCounter]:
                    obs[:,obsix] = sample[:,i]*sample[:,j]
                    obsix += 1
                else:
                    nonVotingPairsIxCounter += 1
                counter += 1
        return obs

    def mch_approximation( samples, dlamda ):
        dE = calc_e_subsystem(samples,dlamda)
        dE -= dE.min()
        ZFraction = 1. / np.mean(np.exp(-dE))
        predsisj = pair_corr( samples, weights=np.exp(-dE)/len(dE) )[1] * ZFraction
        predsisj = np.delete(predsisj,nonVotingPairsIx)
        assert not (np.any(predsisj<-1.00000001) or
            np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(),
                    predsisj.max())
        return predsisj

    return calc_e_subsystem, calc_observables, mch_approximation

def find_non_voting_pairs(sample, threshold=1):
    """
    Find all pairs of voters that did not vote at least a specified threshold number of times.

    Parameters
    ----------
    sample : ndarray
        (n_samples,n_dim). +/-1 basis with 0 for non votes.
    threshold : int
        Least number of votes below which pair is not considered as having voted together.

    Returns
    -------
    list
        Indices of the list of all combinations of pairs that did not vote together determined by
        where either voter's vote was 0.
    """

    from itertools import combinations
    n = sample.shape[1]
    ix = []
    
    counter = 0
    for i,j in combinations(range(n),2):
        if (sample[:,[i,j]].prod(1)!=0).sum()<threshold:
            ix.append( counter )
        counter += 1
    return ix

def all_voting_subsets_of_size_n(n):
    """Return all subsets in majvote that voted together of size n.
    
    Parameters
    ----------
    n : int
    
    Returns
    -------
    list of lists of ints
        Each list specifies a subset of size n that voted together at least once.
    """

    from itertools import combinations

    rowIx = []
    # Iterate through all rows in data.
    for row in majvote:
        thisVoteVoters = np.where(np.isnan(row)==0)[0]
        # If there are at least n voters who voted, iterate through all subsets of size n.
        for combo in combinations(thisVoteVoters,n):
            rowIx.append(combo)

    natCourtsJustices = [list(x) for x in set(tuple(x) for x in rowIx) if len(x)==n]
    return natCourtsJustices

