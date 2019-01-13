# Author: Eddie Lee, edl56@cornell.edu
import numpy as np
from misc.utils import unique_rows
from uuid import uuid4
import pickle
import os


def calc_p_S_given_case(X, issue, issueTable, solver,
                        sample_size=1000):
    """
    Calculate the probability of a super vote for a particular case. This means taking p(S|s) 
    for the subset that we have votes for for that particular issue and then averaging over 
    all s that correspond to the issue, i.e. p(S|case) = sum_s p(S|s)*p(s|case)
    
    Instead of iterating through all states which is much more memory expensive, I just sample
    from p(S|s) to estimate p(S|case).
    
    Parameters
    ----------
    X : ndarray
        The data set that was used to build the maxent model. The nonvotes=0.
    issue : int
        Issue type.
    solver : coniii.solvers.Solver
        This will be used to sample.
    sample_size : int
        Size of the sample.

    Returns
    -------
    ndarray
        Conditional probability of each the voting states returned.
    ndarray
        The unique conditional states that were returned.
    """

    from coniii.utils import state_probs
    
    tmpfile = 'cache/%s.p'%str(uuid4())

    # Get all states that correspond to this issue, but only keep track of the unique states.
    issueix = np.where(issueTable==issue)[0]
    datap, uniqStates = state_probs(X[issueix])
    print("Iterating over %d unique states instead of all %d states."%(len(uniqStates),len(issueix)))
    
    # Iterate through all unique votes for this issue and generate samples from the conditional
    # distribution of the Super Court where the subset s is fixed.
    condSampleP = []
    condSampleUniqStates = []
    toRemoveIx = np.zeros((len(uniqStates)), dtype=bool)
    for i,s in enumerate(uniqStates):
        if np.any(s):
            # Use Bayes' rule for calculation of P(S|s).
            fixedSpins = [(nonzeroix,s[nonzeroix]) for nonzeroix in np.where(s!=0)[0]]
            solver.sampler.generate_cond_samples(sample_size, fixedSpins, burn_in=1000, parallel=True)
            # symmetrize (assuming of course that the given data set has not been symmetrized)
            condSample = np.vstack((solver.sampler.samples, -solver.sampler.samples))

            # for each condition, compress the probability distrib
            uniqinvix = unique_rows(condSample, return_inverse=True)
            uniqix = unique_rows(condSample)
            condp = np.bincount(uniqinvix).astype(float)
            condp /= condp.sum()
            # weight distrib by prob of the condition
            condSampleP.append( condp )
            condSampleUniqStates.append( condSample[uniqix].astype(int) )

            # save results just in case of error
            pickle.dump({'condSampleUniqStates':condSampleUniqStates, 'condSampleP':condSampleP},
                        open(tmpfile,'wb'), -1)
        else:
            # Remove any empty votes
            toRemoveIx[i] = True

    if toRemoveIx.any():
        datap = datap[toRemoveIx==0]
        datap /= datap.sum()
        uniqStates = uniqStates[toRemoveIx==0]
        print("Removed %d states"%toRemoveIx.sum())

    # weight the distributions by the prob of the condition
    condSampleP = [p*datap[i] for i,p in enumerate(condSampleP)]

    # combine cond samples into a single reweighted distribution
    sample = np.vstack(condSampleUniqStates)
    samplep = np.concatenate(condSampleP)
    assert np.isclose(samplep.sum(), 1)
    assert len(sample)==len(samplep)
    
    # compress combined sample by combining probabilities of states that are duplicates
    uinvix = unique_rows(sample, return_inverse=True)
    usample = sample[unique_rows(sample)]

    combsamplep = np.zeros(len(usample))
    for i,ix in enumerate(np.unique(uinvix)):
        combsamplep[i] = samplep[uinvix==ix].sum()
    assert np.isclose(combsamplep.sum(), 1), combsamplep.sum()
    
    # clean up
    os.remove(tmpfile)
    return combsamplep, usample
