# ====================================================================================== #
# Methods for forming blocs of components and for measuring their properties.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *



def p_k(X, weights=None, K=3):
    """From sample of K-Potts states, calculate probability distribution over
    coarse-graining of unique state counts. 

    K=3 is what is used in large_fim.Coupling3.

    Parameters
    ----------
    X : ndarray
    weights : ndarray, None
        Relative weights for each element given in X. This does not have to be normalized
        to one.
    K : int, 3
        3 for 3-state Potts model. Assuming states are as {0,1,2}.
        2 for regular binary Ising model. Assuming states are as {0,1}.

    Returns
    -------
    ndarray
        Probability of seeing a particular binned breakdown.
    ndarray
        Bins.
    """
    
    assert K>1
    assert set(np.unique(X)) <= frozenset((list(range(K))))
   
    counts = np.zeros((len(X), K), dtype=int)
    for k in range(K):
        counts[:,k] = (X==k).sum(1)
    # sort by order so that the only thing that distinguishes rows is the total
    # no. in each bin
    counts = np.sort(counts, axis=1)[:,::-1]

    if weights is None:
        bins, p = np.unique(counts, axis=0, return_counts=True)
        p = p / p.sum()
    else:
        bins, ix, p = np.unique(counts, axis=0, return_counts=True, return_inverse=True)
        summedWeights = np.zeros(p.size)
        for i in range(ix.max()+1):
            summedWeights[i] += weights[ix==i].sum() / (ix==i).sum()
        p = p * summedWeights / p.dot(summedWeights)

    return p, bins

def enumerate_unique_splits(n, K=3):
    """Iterate through unique binnings of neurons for K-state Potts model. See
    count_unique_splits() for more details.

    Parameters
    ----------
    n : int
    K : int, 3
        Number of unique states.

    Returns
    -------
    ndarray
    """
    
    assert K in (2,3)

    splits = np.zeros((count_unique_splits(n, K), K), dtype=int)

    if K==3:
        s = 0  # keep track of arrangements as we iterate through them
        for i in range(n, n//3-(n%3)==0, -1):
            n1 = min(n - i, i)
            for j in range(n1, n1//2-1, -1):
                k = n - i - j
                if k<=j:
                    splits[s] = i, j, k
                    s += 1
    elif K==2:
        counter = 0
        for i in range(n, int(np.ceil(n/2))-1, -1):
            splits[counter] = i, n-i
            counter += 1
    return splits
 
def count_unique_splits(n, K=3):
    """Count number of unique binnings of neurons for k=3 Potts model.

    Imagine binning the neurons into 3 groups. The only thing that matters is how many are
    in each group. Obviously, only the number of neurons in each group matters, not which
    neuron is in which group. Furthermore, we only care about the numbers in each
    partition, and not the order in which the partitions are lined up.
    
    Thus, we can imagine taking the largest plurality first. This must be equal to or
    larger than N/3. Then, the second group must be equal to or larger than the last
    group. So we must iterate through N-i down to (N-i)/2. The number in the last
    partition is given by the previous two partitions.
    
    Parameters
    ----------
    n : int

    Returns
    -------
    int
    """
    
    assert K in (2,3)

    if K==2:
        return int(n//2) + 1

    elif K==3:
        s = 0  # keep track of arrangements as we iterate through them
        for i in range(n, n//3-(n%3)==0, -1):
            n1 = min(n - i, i)
            for j in range(n1, n1//2-1, -1):
                k = n - i - j
                if k<=j:
                    s += 1
        return s

def p_maj(X, weights=None):
    """Coarse-grained probability distribution only considering the probability of k votes
    in the plurality.

    Parameters
    ----------
    X : ndarray
    weights : ndarray, None
        Relative weights for each element given in X. This does not have to be normalized
        to one.

    Returns
    -------
    ndarray
        Probability of seeing a particular binned breakdown.
    ndarray
        Bins.
    """
    
    counts = np.array([np.bincount(row).max() for row in X])

    if weights is None:
        bins, p = np.unique(counts, return_counts=True)
        p = p / p.sum()
    else:
        bins, ix, p = np.unique(counts, return_counts=True, return_inverse=True)
        summedWeights = np.zeros(p.size)
        for i in range(ix.max()+1):
            summedWeights[i] += weights[ix==i].sum() / (ix==i).sum()
        p = p * summedWeights / p.dot(summedWeights)

    return p, bins
