# ============================================================================================ #
# Quick access to useful modules from SCOTUS project.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ============================================================================================ # 
import numpy as np
from numba import njit
from coniii.utils import *
from .fim import IsingFisherCurvatureMethod2  # for compatibility with old pickles
import importlib
from warnings import warn
from itertools import combinations
np.seterr(divide='ignore')


# ========= #
# Functions #
# ========= #
def fisher_subspace(n, result, rtol=.05):
    """Wrapper for extracting individual subspace eigenvalues.

    Parameters
    ----------
    n : int
    result : tuple
        Elements saved in fisherResult.
    rtol : float, .05

    Returns
    -------
    ndarray
    """

    isingdkl, (hess, errflag, err), eigval, eigvec = result
    
    if np.linalg.norm(err)<(rtol*np.linalg.norm(hess)):
        # when limited to the subspace of a single justice at a given time (how do we 
        # optimally tweak a single justice to change the system?)
        justiceEigval = []
        justiceEigvec = []

        for j in range(n):
            subspaceHess = hess[j*(n-1):(j+1)*(n-1),j*(n-1):(j+1)*(n-1)]
            u,v = np.linalg.eig(subspaceHess)
            sortix = np.argsort(u)[::-1]
            u = u[sortix]
            v = v[:,sortix]

            justiceEigval.append(u)
            justiceEigvec.append(v)
        justiceEigval = np.vstack(justiceEigval)

        primaryEigval = eigval[0]
        topVoterEigvals = np.sort(justiceEigval[:,0])[::-1][:3]
        assert (topVoterEigvals[0]/primaryEigval)<=1, "Individual subspace eigenvector is larger."
    else:
        raise Exception("Error on Hessian is large.")
    
    return primaryEigval, topVoterEigvals

def multi_info(X, hJ, n_boot=10_000, disp=True, **Squad_kwargs):
    """
    Parameters
    ----------
    X : ndarray
    hJ : ndarray
    n_boot : int, 10_000
    disp : bool, True

    Returns
    -------
    float
        Multi-information captured.
    tuple
        (Sdata, Sind, Spair)
    ndarray
        fit
    """

    from entropy.estimators import S_quad
    import importlib
    
    # estimate the data entropy (enforcing symmetrization of prob)
    Sdata, fit, err = S_quad(np.unique(X, axis=0, return_counts=True)[1],
                             np.logspace(.5,1,10), n_boot,
                             X_is_count=True, parallel=True, return_fit=True,
                             **Squad_kwargs)
    
    # measure pairwise model entropy
    ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%X.shape[1])
    p = ising.p(hJ)
    
    Spair = -p.dot(np.log2(p))
    
    Sind = X.shape[1]
    if disp and Spair<Sdata:
        print(Sdata,Sind,Spair)
    return 1-(Spair-Sdata)/(Sind-Sdata), (Sdata, Sind, Spair), fit

def check_correlations(X, p, orders, allStates=None):
    """Check how well correlations are fit by model probability distribution.
    
    Parameters
    ----------
    X : ndarray
        Data samples. Assuming {-1,1} basis.
    p : ndarray
        Probability distribution of all possible states given model.
    orders : list
        Correlation functions of orders to check.
    allStates : ndarray, None
    
    Returns
    -------
    ndarray
        Errors for each correlation order given.
    ndarray
        Correlation coefficient.
    """
    
    from coniii import bin_states

    if allStates is None:
        allStates = bin_states(X.shape[1], sym=True)

    errs = np.zeros(len(orders))
    corr = np.zeros(len(orders))

    for i,k in enumerate(orders):
        Xcorr = k_corr(X, k)
        modelcorr = k_corr(allStates, k, weights=p)
        errs[i] = np.linalg.norm(Xcorr-modelcorr)
        corr[i] = np.corrcoef(Xcorr, modelcorr)[0,1]
    return errs, corr
    
def coarse_grain(X, nbins, sortix=None, method='maj', params=()):
    """Coarse-grain given votes into n bins by using specified coarse-graining method. If
    method is majority and a vote is tied, a random vote is chosen.
    
    Parameters
    ----------
    X : ndarray
        (n_votes, n_voters) in {-1,1} basis.
    nbins : int
    sortix : ndarray, None
        By default, votes will be sorted in default order.
    method : str, 'maj'
        'maj', 'anypos', or 'corr'
    params : tuple, ()
        Parameters specifying details of coarse-graining.
    
    Returns
    -------
    coarseX
    """
    
    assert set(np.unique(X))<=frozenset((-1,0,1))
    if sortix is None:
        sortix = np.arange(X.shape[1], dtype=int)
    
    binsix = []  # indices of cols in each bin

    if method=='maj':
        # find locations of where the index cutoffs will be per bin
        bins = np.linspace(0, sortix.size-1, nbins+1)
        bins[-1] += np.inf
        
        # sort into bins
        groupix = np.digitize(sortix, bins)

        coarseX = np.zeros((len(X), nbins), dtype=int)
        for i in range(1,nbins+1):
            binsix.append( np.where(groupix==i)[0] )
            coarseX[:,i-1] = np.sign(X[:,groupix==i].sum(1))
        coarseX[coarseX==0] = np.random.choice([-1,1], size=(coarseX==0).sum())
    elif method=='anypos':
        # find locations of where the index cutoffs will be per bin
        bins = np.linspace(0, sortix.size-1, nbins+1)
        bins[-1] += np.inf
        
        # sort into bins
        groupix = np.digitize(sortix, bins)

        coarseX = np.zeros((len(X), nbins), dtype=int)
        for i in range(1,nbins+1):
            binsix.append( np.where(groupix==i)[0] )
            coarseX[:,i-1] = (X[:,groupix==i]==1).any(1) * 2 - 1
        coarseX[coarseX==0] = np.random.choice([-1,1], size=(coarseX==0).sum())
 
    elif method=='corr':
        from scipy.spatial.distance import squareform
        coarseX = X.copy()
        originalIx = [[i] for i in range(X.shape[1])]

        # Combine sets of spins with the largest pairwise correlations
        while coarseX.shape[1]>nbins:
            n = coarseX.shape[1]
            cij = squareform(pair_corr(coarseX)[1])
            ix = list(range(coarseX.shape[1]))
            
            newClusters = []
            for i in range(n//2):
                # find maximally correlated pair of spins
                mxix = np.argmax(cij.ravel())
                mxix = (mxix//(n-2*i), mxix-(n-2*i)*(mxix//(n-2*i)))  # row and col
                if mxix[0]>mxix[1]:
                    mxix = (mxix[1],mxix[0])
                
                newClusters.append((ix[mxix[0]], ix[mxix[1]]))
                # remove corresponding rows and cols of pair
                cij = np.delete(np.delete(cij, mxix[0], axis=0), mxix[0], axis=1)
                cij = np.delete(np.delete(cij, mxix[1]-1, axis=0), mxix[1]-1, axis=1)
                ix.pop(mxix[0])
                ix.pop(mxix[1]-1)
                #print(cij.shape, len(ix))
            if n%2:
                # if X contains an odd number of voters
                newClusters.append((ix[0],))
            # check that every index appears once (and only once)
            assert np.array_equal(np.sort(np.concatenate(newClusters)),np.arange(n)), newClusters
            
            # coarse-grain votes such any positive lead to positive vote
            X_ = np.zeros((coarseX.shape[0],int(np.ceil(n/2))), dtype=int)
            originalIx_ = []
            for i,ix in enumerate(newClusters):
                X_[:,i] = (coarseX[:,ix]==1).any(1)*2-1
                originalIx_.append([])
                for ix_ in ix:
                    originalIx_[-1] += originalIx[ix_]
            originalIx = originalIx_
            coarseX = X_
        binsix = originalIx
    else:
        raise NotImplementedError("Invalid method option.")

    return coarseX, binsix

def ndarray2text(X, fname, cols=None):
    """Export ndarray to a simple text file for use with R. Assumes that entries are int.
    
    Parameters
    ----------
    X : ndarray
    fname : str
    cols : list of str
    """
    
    import gzip
    if not cols is None:
        assert len(cols)==X.shape[1]
    
    with gzip.open(fname, 'w') as f:
        if not cols is None:
            f.write((' '.join(cols)+'\n').encode('utf-8'))
        
        for row in X:
            f.write((' '.join(['%d'%i for i in row.tolist()])+'\n').encode('utf-8'))

def remove_principal_mode(X):
    """Remove principal eigenvector dimension from row space of matrix. For example, you
    might use this to look beyond the collective mode in some cases.

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    ndarray
    """
    
    # identify principal mode
    el, v = np.linalg.eig(X)
    ix = np.argmax(el)
    v = v[:,ix].real

    # remove from row space
    X = X - (X.dot(v)[:,None]*v[None,:])

    return X

def solve_linearized_pair_perturbation(n, iStar, aStar, sisj, p,
                                       allStates=None,
                                       eps=1e-6,
                                       check_stability=True,
                                       full_output=False):
    """Only considering pairs as <si*sj> is converted into <si*sk>.
    
    Find linear perturbation for couplings when spin iStar is fractionally converted
    into spin aStar. This is equivalent to a linear transformation of every one of iStar's
    correlations into aStar's correlations at small rate eps. Or sampling randomly iStar's
    orientation and replacing with aStar's with probability eps.
    
    Solve following system of equations.
    A.dot(delta_{kl}) = C
    
    Parameters
    ----------
    n : int
    iStar : int
        Index of spin to be replaced.
    aStar : int
        Index of spin to replace with.
    sisj : ndarray
        Pairwise correlations.
    p : ndarray
    allStates : ndarray, None
        {-1,1} basis.
    eps : float, 1e-6
        Step size for approximating derivative.
    check_stability : bool, True
        Check whether or not linearization is accurate by halving eps.
    full_output : False
    
    Returns
    -------
    ndarray
        Jacobian of couplings wrt eps parameter titrating votes.
    """
    
    assert sisj.size==(n*(n-1)//2)
    if allStates is None:
        allStates = bin_states(n, True)

    A = np.zeros((n*(n-1)//2,n*(n-1)//2))
    C = np.zeros(n*(n-1)//2)
    for ijCount,(i,j) in enumerate(combinations(range(n),2)):
        if i==iStar:
            if aStar<j:
                C[ijCount] = eps*(sisj[unravel_index((aStar,j),n)] - sisj[unravel_index((i,j),n)])
            elif aStar>j:
                C[ijCount] = eps*(sisj[unravel_index((j,aStar),n)] - sisj[unravel_index((i,j),n)])
            else:
                C[ijCount] = eps*(1-sisj[unravel_index((i,j),n)])
        elif j==iStar:
            if aStar<i:
                C[ijCount] = eps*(sisj[unravel_index((aStar,i),n)] - sisj[unravel_index((i,j),n)])
            elif aStar>i:
                C[ijCount] = eps*(sisj[unravel_index((i,aStar),n)] - sisj[unravel_index((i,j),n)])
            else:
                C[ijCount] = eps*(1-sisj[unravel_index((i,j),n)])
        # this is just a copy of sisj if nothing changes
        newsisj = C[ijCount] + sisj[unravel_index((i,j),n)]

        for klCount,(k,l) in enumerate(combinations(range(n),2)):
            A[ijCount,klCount] = np.prod(allStates[:,(i,j,k,l)],1).dot(p) - newsisj*sisj[klCount]
    
    # factor out linear dependence on eps
    dJ = np.linalg.solve(A,C)/eps

    if check_stability:
        # double epsilon and make sure solution does not change by a large amount
        dJtwiceEps = solve_linearized_pair_perturbation(n, iStar, aStar, sisj, p,
                                                        eps=eps/2,
                                                        allStates=allStates,
                                                        check_stability=False)
        # print if relative change is more than .1% for any entry
        if ((np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ)))>-3).any():
            print("Unstable solution. Recommend shrinking eps.")
    if full_output:
        return dJ, (A, C)
    return dJ

def tweak_constraints(constraints, i, j, theta, n):
    """Replace all of i's correlations with j's correlations with weighting theta.
    
    Parameters
    ----------
    constraints : ndarray
        Concatenated list of h and J.
    i : int
        Spin to replace.
    j : int
        Spin to replace with.
    theta : ndarray
        Probability of replacement.
    n : int
        System size.
        
    Returns
    -------
    ndarray
        Updated constraints.
    """
    
    from coniii.utils import unravel_index
    newconstraints = constraints.copy()
    
    # means
    newconstraints[i] = constraints[j]*theta + constraints[i]*(1-theta)
    
    # pairwise correlations
    for k in np.delete(range(n), [i,j]):
        if i<k:
            ik=unravel_index((i,k),n)
        else:
            ik=unravel_index((k,i),n)
        if j<k:
            jk=unravel_index((j,k),n)
        else:
            jk=unravel_index((k,j),n)
        if i<j:
            ij=unravel_index((i,j),n)
        else:
            ij=unravel_index((j,i),n)
        
        newconstraints[ik+n]=constraints[ik+n]*(1-theta) + theta*constraints[jk+n]
        newconstraints[ij+n]=constraints[ij+n]*(1-theta) + theta
    return newconstraints

def define_energy_basin_functions(calc_observables):
    """Define functions for finding energy basins. Need calc_observables() to be defined in global namespace.

    Parameters
    ----------
    calc_observables : function

    Returns
    -------
    function
        find_energy_basin
    function
        flip_least_stable_spin
    """

    def find_energy_basin(x, hJ):
        """
        Find the local energy minimum.
        """

        x = x.copy()
        xprev = np.zeros(len(x), dtype=int)
        while (x!=xprev).any():
            xprev = x.copy()
            x = flip_least_stable_spin(x, hJ)
        return x
    
    def flip_least_stable_spin(x, hJ, calc_observables=calc_observables):
        """
        Flip the least stable spin.
        """

        E = -calc_observables(x[None,:]).dot(hJ)
        dE = np.zeros(x.size)
        for i in range(x.size):
            x[i] *= -1
            dE[i] = -calc_observables(x[None,:]).dot(hJ)-E
            x[i] *= -1

        if (dE<0).any():
            x[np.argmin(dE)] *= -1
        return x 

    return find_energy_basin, flip_least_stable_spin
