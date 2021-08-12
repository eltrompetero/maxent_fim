# ====================================================================================== #
# Quick access to useful modules for pivotal components projects.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from numba import njit
import importlib
from warnings import warn
from itertools import combinations, product
import os
import dill as pickle
from threadpoolctl import threadpool_limits
from multiprocess import Pool, cpu_count, set_start_method
import mpmath as mp
from scipy.optimize import minimize

from coniii.utils import *

from .fim import Coupling  # for compatibility with old pickles

np.seterr(divide='ignore')



# ========= #
# Functions #
# ========= #
def fit_decay_power_law(y, auto_upper_cutoff=0.):
    """Fit an exponentially truncated power law.

    C * x^-alpha * exp(-l * x)
    
    Parameters
    ----------
    y : ndarray
        Rank-ordered eigenvalues.
    auto_upper_cutoff : float, 0.
        If nonzero, then fit does not account for points at tail that decrease faster than
        this value.
        
    Returns
    -------
    lambda fcn
    dict
        As is returned from scipy.optimize.minimize
        Parameters correspond to  C, alpha, log(l)
    """
    
    # setup
    assert (np.diff(y)<=0).all(), "Eigenvalues should be rank-ordered."
    assert auto_upper_cutoff<=0

    if auto_upper_cutoff:
        x = np.arange(1, y.size+1)

        # find the first point from the end that doesn't violate cutoff
        slope = np.diff(np.log(y))/np.diff(np.log(x))
        ix = np.argmax(slope[::-1]>auto_upper_cutoff)

        if ix!=0:
            assert ix<(y.size-1), ix
            y = y[:-ix]
            x = np.arange(1, y.size+1)
    
    # define log cost
    def cost(params):
        c, alpha, el = params  # offset, pow law exp, cutoff scale
        return np.linalg.norm(np.log(c) -alpha * np.log(x) - np.exp(el) * x - np.log(y))
    
    # find optimal parameters (this may need some adjusting if fit is not good)
    soln = minimize(cost, [y[0], 1.2, 20],
                    bounds=[(1, np.inf), (.2, 4), (-np.inf,np.inf)])
    
    # determine plotting function
    c, alpha, el = soln['x']
    el = np.exp(el)
    fit_fun = lambda x, c=c, alpha=alpha, el=el: c * x**-alpha * np.exp(-el * x)

    return fit_fun, soln

def fields_vec2mat(h, K=3):
    """Take vector of fields organized as N fields at a time (for each state), into a
    matrix of K columns.

    Parameters
    ----------
    h : ndarray
    K : int, 3

    Returns
    -------
    ndarray
    """

    assert h.ndim==1
    N = h.size//K
    assert h.size/K==N
    
    hmat = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            hmat[i,k] = h[i+N*k]

    return hmat

def coarse_grain_potts3(all_states, no):
    """Function for coarse-graining 3-state Potts spins. 

    Parameters
    ----------
    all_states : ndarray
        (n_samples, n_spins)
    no : int
        Number of coarse-graining function to employ.
        1) Number of spins in each of the three possible states (without regard for the
        number for any particular state). This means that four spins partitioned into
        occupancy numbers (0,2,2) is the same as (2,2,0).
        2) Number of spins in the plurality only. 

    Returns
    -------
    ndarray
        Number of votes in each respective state ordered from max to min.
    """
    
    if no==1:
        kVotes = list(map(lambda x:np.sort(np.bincount(x, minlength=3))[::-1],
                          all_states))
    elif no==2:
        kVotes = list(map(lambda x:np.bincount(x).max(),
                          all_states))
    else:
        raise NotImplementedError

    return np.vstack(kVotes)

def perturb_3_spin(p, k,
                   eps=1e-4,
                   return_delta=False,
                   run_checks=False):
    """Perturb configuration of 3-state spin along the shortest distance in triangular
    convex hull to target state.

    We use the replacement rule pi -> pi * (1-eps) + eps. The remaining other two
    probabilities are modified by calculating the projection along the respective
    probability axes. These axes are given by the shortest distance to the corresponding
    edge.

    Code was originally written for particular case of perturbaiton towards p[1], but
    uses rolled axes to consider all other dimensions.

    Parameters
    ----------
    p : ndarray
        Probability as 3-vector.
    k : int
        Dimension along which to effect primary perturbation. 0<=k<=2.
    eps : float, 1e-4
        Strength of perturbation.
    return_delta : bool, False
    run_checks : bool, False
    
    Returns
    -------
    ndarray
        Either new probabilities of configurations after perturbation or the delta that
        needs to be added to each element of old probability vector.
    """
    
    if run_checks: assert np.isclose(p.sum(), 1) and p.size==3 and (p>=0).all()
    assert 0<=k<=2
    
    if k==0:
        rollno = 2
    elif k==2:
        rollno = 1
    else:
        rollno = 0
    p = np.roll(p, -rollno)
    
    xy = (p[1] * 2 + p[2]) / np.sqrt(3), p[2]
    
    r = np.array([np.linalg.norm(xy),
                  np.linalg.norm([xy[0]-2/np.sqrt(3), xy[1]]),
                  np.linalg.norm([xy[0]-1/np.sqrt(3), xy[1]-1])])
    
    # account for special case where one of the states as p=1
    if (r==0).any():
        zeroix = np.where(r==0)[0][0]
        r[zeroix] = 1
        theta = np.arcsin(p[[2,0,1]]/r)
        theta[zeroix] = np.pi/6
    else:
        theta = np.arcsin(p[[2,0,1]]/r)
    
    # this can only be the case if the hypotenuse is shorter than a leg!
    if run_checks: assert ~np.isnan(theta).any()

    deltanorm = eps * (1-p[1]) / np.cos(theta[1]-np.pi/6)
    delta = np.array([np.cos(np.pi/2-theta[1]),
                      np.cos(theta[1]-np.pi/6),
                      np.cos(np.pi/6+theta[1])])
    
    if run_checks: assert np.isclose(delta[1] - delta[0] - delta[2], 0), delta
    
    if return_delta:
        d = deltanorm * delta * np.array([-1,1,-1])
        return np.roll(d, rollno)
 
    newp = p + deltanorm * delta * np.array([-1,1,-1])
    newp = np.roll(newp, rollno)
    
    return newp

def block_mean_fim(n, fim):
    """Coarse grain FIM by taking averages of blocks that correspond to perturbations
    focused on particular receiver and target pairs.
    
    Parameters
    ----------
    n : int
    fim : ndarray
    
    Returns
    -------
    ndarray
        Coarse-grained FIM.
    """
    
    assert n==(fim.shape[0]/(n-1))
    coarsefim = np.zeros((n,n))
    
    # coarse-grain diagonal
    for i in range(n):
        coarsefim[i,i] = fim[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)].mean()
    # coarse grain off-diagonal elements
    for i in range(n-1):
        for j in range(i+1, n):
            coarsefim[i,j] = coarsefim[j,i] = fim[i*(n-1):(i+1)*(n-1),j*(n-1):(j+1)*(n-1)].mean()
            
    return coarsefim

def block_sum_fim(n, fim):
    """Coarse grain FIM by taking sums of blocks that correspond to perturbations focused
    on particular receiver and target pairs.
    
    Parameters
    ----------
    n : int
    fim : ndarray
    
    Returns
    -------
    ndarray
        Coarse-grained FIM.
    """
    
    assert n==(fim.shape[0]/(n-1))
    coarsefim = np.zeros((n,n))
    
    # coarse-grain diagonal
    for i in range(n):
        coarsefim[i,i] = fim[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)].sum()
    # coarse grain off-diagonal elements
    for i in range(n-1):
        for j in range(i+1, n):
            coarsefim[i,j] = coarsefim[j,i] = fim[i*(n-1):(i+1)*(n-1),j*(n-1):(j+1)*(n-1)].sum()
            
    return coarsefim

def missing_fim_files(dr, mnix, mxix):
    """Display names of files missing in the sequence.

    Parameters
    ----------
    dr : str
    mnix : int
    mxix : int
    
    Returns
    -------
    """

    files = os.listdir(dr)
    nums = sorted([int(f.split('_')[1]) for f in files])

    missingnums = []
    for i in range(mnix, mxix+1):
        if not i in nums:
            missingnums.append(i)
    return missingnums

def disconnected_components(adj):
    """Identify each independent component using matrix multiplication walk.
    
    Parameters
    ----------
    adj : ndarray
        Square matrix.
    
    Returns
    -------
    list of lists
    """
    
    nodes = list(range(adj.shape[0]))
    adj = adj.copy()
    adj[np.diag_indices_from(adj)] = 1
    walk = np.linalg.matrix_power(adj, adj.shape[0])
    components = []
    
    while nodes:
        components.append([])
        
        start = np.zeros(adj.shape[0])
        start[nodes[0]] = 1
        end = walk.dot(start)
        
        for n in np.where(end)[0].tolist():
            components[-1].append(nodes.pop(nodes.index(n)))
    
    return components

def largest_component(adj):
    """Return largest connected component.

    Parameters
    ----------
    adj : ndarray
        Square matrix.
    
    Returns
    -------
    list
    """

    components = disconnected_components(adj)
    ix = np.argmax([len(c) for c in components])
    return components[ix]

def match_mismatched_p(*args, bins=None):
    """Align two probability distributions that are over different states.

    Parameters
    ----------
    twople of (bins, p)
        bins : ndarray
            Unique states to which each probability in p1 corresponds to.
        p : ndarray
            First probability distribution to compare with second.
    bins : ndarray, None
        Option to pass in bins into which to bin probabilities. Assertion error will be
        thrown if it does not contain all possibilities aggregated across given bins.

    Returns
    -------
    list of ndarray
        New probability distributions.
    ndarray
        New bins.
    """
    
    if args[0][0].ndim==1:
        args = [(b[:,None], p) for b, p in args]
        bins = bins[:,None]
    elif args[0][0].ndim!=2:
        raise NotImplementedError

    n = []
    for b, p in args:
        assert b.ndim==2
        assert b.shape[0]==p.size
        n.append(p.size)
    
    if bins is None:
        bins, ix = np.unique(np.concatenate([b for b, p in args], axis=0),
                             return_inverse=True, axis=0)

        ncum = np.cumsum(n)
        ix = [ix[ncum[i]:ncum[i+1]] for i in range(len(args)-1)]
        ix.insert(0, 0)
    else:
        # TODO: a very slow looping method could be sped up?
        assert bins.ndim==2
        # assuming that bins consists of unique elements, we can use hash lookup to
        # speed up the search
        binsdict = dict([(tuple(b.tolist()), i) for i, b in enumerate(bins)])

        ix = []
        for i, (b, p) in enumerate(args):
            ix.append(np.zeros(p.size, dtype=int))
            for j, x in enumerate(b):
                ix[i][j] = binsdict[tuple(x.tolist())]

    newp = [np.zeros(bins.shape[0]) for i in range(len(args))]

    for i, b in enumerate(bins):  # iterate thru each element in concat bins
        for j, (oldbins, oldp) in enumerate(args):  # fill in values for expanded prob distributions
            matchix = (b[None,:]==oldbins).all(1)
            if matchix.any():
                newp[j][i] = oldp[matchix]

    assert all([np.isclose(p.sum(), 1) for p in newp])
    
    if bins.shape[1]==1:
        bins = bins.ravel()
    return newp, bins

def vec2mat(vec):
    """Reshape eigenvector detailing results of pairwise perturbations into a matrix. This
    is also transposed such that the columns correspond to local perturbations and the
    rows correspond to neighborhood perturbations as is the preferred method of analysis.

    Parameters
    ----------
    vec : ndarray

    Returns
    -------
    ndarray
        Pairwise perturbation matrix from FIM eigenvector.
    """

    n = (1+np.sqrt(1+4*vec.size)) / 2
    assert int(n)==n, "Cannot be reshaped into n,n pairwise matrix with zeroed diagonal."
    n = int(n)

    return np.insert(vec, range(0, n*n, n), 0).reshape(n,n).T

def load_Mag3(fname):
    """Load a model from large_fim that has been pickled. Regular pickling routine does
    not work for them!

    Parameters
    ----------
    fname : str
        name of pickled file

    Returns
    -------
    large_fim.Mag3
    """
    
    from .large_fim import Mag3
    assert os.path.isfile(fname)

    # load model. must initialize a template instance first
    n = 5
    h = np.concatenate((np.random.normal(size=n*2, scale=.5), np.zeros(n)))
    J = np.random.normal(size=n*(n-1)//2, scale=1/n)
    model = Mag3(n, h, J, n_samples=100, eps=1e-4, precompute=False, iprint=False)
    state = pickle.load(open(fname, 'rb'))
    model.__set_state__(state) 
    n = model.n
    
    return model

def load_CanonicalMag3(fname):
    """Load a model from large_fim that has been pickled. Regular pickling routine does
    not work for them!

    Parameters
    ----------
    fname : str
        name of pickled file

    Returns
    -------
    large_fim.CanonicalMag3
    """
    
    from .large_fim import CanonicalMag3
    assert os.path.isfile(fname)

    # load model. must initialize a template instance first
    n = 5
    h = np.concatenate((np.random.normal(size=n*2, scale=.5), np.zeros(n)))
    J = np.random.normal(size=n*(n-1)//2, scale=1/n)
    model = CanonicalMag3(n, h, J, n_samples=100, eps=1e-4, precompute=False, iprint=False)
    state = pickle.load(open(fname, 'rb'))
    model.__set_state__(state) 
    n = model.n
    
    return model

def load_CanonicalCoupling3(fname):
    """Load a model from large_fim that has been pickled. Regular pickling routine does
    not work for them!

    Parameters
    ----------
    fname : str
        name of pickled file

    Returns
    -------
    large_fim.CanonicalCoupling3
    """
    
    from .large_fim import CanonicalCoupling3
    assert os.path.isfile(fname)

    # load model. must initialize a template instance first
    n = 5
    h = np.concatenate((np.random.normal(size=n*2, scale=.5), np.zeros(n)))
    J = np.random.normal(size=n*(n-1)//2, scale=1/n)
    model = CanonicalCoupling3(n, h, J, n_samples=100, eps=1e-4, precompute=False, iprint=False)
    state = pickle.load(open(fname, 'rb'))
    model.__set_state__(state)
    n = model.n
    
    return model

def load_Coupling3(fname):
    """Load a model from large_fim that has been pickled. Regular pickling routine does
    not work for them!

    Parameters
    ----------
    fname : str
        name of pickled file

    Returns
    -------
    large_fim.Coupling3
    """
    
    from .large_fim import Coupling3
    assert os.path.isfile(fname)

    # load model. must initialize a template instance first
    n = 5
    h = np.concatenate((np.random.normal(size=n*2, scale=.5), np.zeros(n)))
    J = np.random.normal(size=n*(n-1)//2, scale=1/n)
    model = Coupling3(n, h, J, n_samples=100, eps=1e-4, precompute=False, iprint=False)
    state = pickle.load(open(fname,'rb'))
    model.__set_state__(state) 
    n = model.n
    
    return model

def combine_fim_files(*args, use_pickle5=True):
    """Combine calculations of FIM from multiple different files.

    Parameters
    ----------
    *args : str
        Each the name of a pickle with an 'fim' variable.
    use_pickle5 : bool, True

    Returns
    -------
    ndarray
        FIM.
    bool
        True if everything works out fine.
    """
    
    if not len(args):
        return np.zeros((0,0)), False
    if use_pickle5:
        import pickle5 as pickle

    success = True
    
    for i, f in enumerate(args):
        assert os.path.isfile(f), f'Missing file {f}.'
        thisfim = pickle.load(open(f,'rb'))['fim']

        if i==0:
            fim = thisfim
        else:
            count = (fim[thisfim!=0]!=0).sum()
            if count>fim.shape[0]: 
                warn("%d off-diagonal entries appear twice."%count)
                success = False
            if not np.array_equal(thisfim.diagonal(), fim.diagonal()):
                err = np.linalg.norm(thisfim.diagonal() - fim.diagonal())
                warn("Diagonals do not match. Mismatch error of %f with file %s."%(err,f))
                success = False
            
            # copy in elements
            fim[thisfim!=0] = thisfim[thisfim!=0]
    
    if (fim==0).any():
        warn("Not every entry filled.")
        success = False

    return fim, success

def refine_maxent_solution(n, sisj, J0,
                           full_output=True,
                           tol=1e-10,
                           max_iter=1000,
                           multiplier=.2):
    """Iterative refinement of couplings.
    
    Parameters
    ----------
    n : int
    sisj : ndarray
        Only pairwise correlations.
    J0 : ndarray
    full_output : bool, True
    tol : float, 1e-10
    max_iter : int, 1000
    multiplier : float, .2

    Returns
    -------
    ndarray
        Couplings.
    list
        Error history.
    """
    
    assert sisj.size==J0.size
    hJ = np.concatenate((np.zeros(n), J0))
    ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
    
    counter = 0
    err_history = [1]
    while counter<max_iter and err_history[-1]>tol:
        dhJ = ising.calc_observables(hJ)[n:] - sisj
        err_history.append(np.linalg.norm(dhJ))
        hJ[n:] -= dhJ*multiplier
        counter += 1
    err_history.pop(0)

    if err_history[-1]<tol:
        errflag = 0
    else:
        errflag = 1
    
    if full_output:
        return hJ[n:], errflag, err_history
    return hJ[n:]

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

def multi_info(X, hJ, method, n_boot=10_000, disp=True, **Spoly_kwargs):
    """Fraction of multi-information captured.

    Parameters
    ----------
    X : ndarray
        Data.
    hJ : ndarray
        Maxent parameters.
    method : str
        IsingFisherCurvatureMethod used.
    n_boot : int, 10_000
    disp : bool, True
    **Spoly_kwargs

    Returns
    -------
    float
        Multi-information captured.
    tuple
        (Sdata, Sind, Spair)
    ndarray
        fit
    """

    from entropy.estimators import S_poly
    import importlib
    
    # estimate the data entropy (enforcing symmetrization of prob)
    Sdata, fit, err = S_poly(np.unique(X, axis=0, return_counts=True)[1],
                             np.logspace(.5,1,10), n_boot,
                             X_is_count=True, parallel=True, return_fit=True,
                             **Spoly_kwargs)
    
    # measure pairwise model entropy
    if not '4' in method:
        ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%X.shape[1])
    else:
        ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_potts'%X.shape[1])
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
        groupix = np.digitize(np.arange(X.shape[1]), bins)

        coarseX = np.zeros((len(X), nbins), dtype=int)
        for i in range(1,nbins+1):
            binsix.append( np.where(groupix==i)[0] )
            coarseX[:,i-1] = np.sign(X[:,groupix==i].sum(1))

    elif method=='anypos':
        # find locations of where the index cutoffs will be per bin
        bins = np.linspace(0, sortix.size-1, nbins+1)
        bins[-1] += np.inf
        
        # sort into bins
        groupix = np.digitize(np.arange(X.shape[1]), bins)

        coarseX = np.zeros((len(X), nbins), dtype=int)
        for i in range(1,nbins+1):
            binsix.append( np.where(groupix==i)[0] )
            coarseX[:,i-1] = (X[:,groupix==i]==1).any(1) * 2 - 1
 
    elif method=='corr':
        # no sortix used here
        raise NotImplementedError
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

def degree_collective_exp(rate, n=9):
    """Example of entropy of the individual subspaces changes for a simple model where
    each subspace is simply a single direction in Cartesian coordinates.
    
    I construct a diagonal matrix with exponentially decaying diagonal entries. These
    correspond to the block subspaces of individual perturbations if each subspace were
    truly independent of the others. In this simplified case, it becomes straightforward
    to consider how the entropy over the subspace eigenvalues changes after I remove the
    principal mode (since the eigenvalues are just the eigenvalues of the diagonal
    matrix).
    
    Parameters
    ----------
    rate : float
    n : int, 9
    
    Returns
    -------
    float
    float
    """
    
    X = np.zeros((n,n))
    # X[eye(len(X))==1] = 2.**(-4*arange(len(X)))
    X[np.eye(len(X))==1] = np.exp(-rate*np.arange(len(X)))

    eigval,_ = np.linalg.eig(X)
    assert (eigval>=0).all()

    p = eigval/eigval.sum()
    r1 = -p.dot(np.log2(p))/np.log2(len(p))

    Xp = remove_principal_mode(X)

    eigval,_ = np.linalg.eig(Xp)
    assert (eigval>=0).all()

    p = eigval/eigval.sum()
    r2 = np.nansum(-p*np.log2(p))/np.log2(p.size)
    
    return r1,r2
