# ====================================================================================== #
# Analysis of features of FIM.
# 
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from scipy.special import binom

from .utils import *



def uniformity(eigvec, axis=0):
    """Uniformity across either rows or columns of FIM.
    
    Parameters
    ----------
    eigvec : ndarray
        Eigenvectors of FIM ordered in the usual way by column.
    axis : int, 0   
        Either 0 or 1 corresponding to col or row uniformity.

    Returns
    -------
    ndarray
    """
    
    from .coarse_grain import count_unique_splits
    from .utils import vec2mat

    # get number of components for eigenvector of pairwise perturbations
    n = (1+np.sqrt(1+4*eigvec.shape[0])) / 2
    assert int(n)==n, "Cannot be reshaped into (n,n) pairwise matrix with zeroed diagonal."
    n = int(n)

    mats = []
    for i in range(count_unique_splits(n)):
        mats.append( vec2mat(eigvec[:,i]) )
    return np.array([(m.sum(axis)**2).sum() for m in mats])

def block_mean(n, X):
    """Coarse grain matrix by taking averages of blocks that correspond to perturbations
    focused on particular receiver and target pairs.
    
    Parameters
    ----------
    n : int
    X : ndarray
    
    Returns
    -------
    ndarray
        Coarse-grained X.
    """
    
    assert n==(X.shape[0]/(n-1))
    coarseX = np.zeros((n,n))
    
    # coarse-grain diagonal
    for i in range(n):
        coarseX[i,i] = X[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)].mean()
    # coarse grain off-diagonal elements
    for i in range(n-1):
        for j in range(i+1, n):
            coarseX[i,j] = coarseX[j,i] = X[i*(n-1):(i+1)*(n-1),j*(n-1):(j+1)*(n-1)].mean()
            
    return coarseX

def block_sum(n, X):
    """Coarse grain matrix by taking sums of blocks that correspond to perturbations
    focused on particular receiver and target pairs.
    
    Parameters
    ----------
    n : int
    X : ndarray
    
    Returns
    -------
    ndarray
        Coarse-grained X.
    """
    
    assert n==(X.shape[0]/(n-1))
    coarseX = np.zeros((n,n))
    
    # coarse-grain diagonal
    for i in range(n):
        coarseX[i,i] = X[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)].sum()
    # coarse grain off-diagonal elements
    for i in range(n-1):
        for j in range(i+1, n):
            coarseX[i,j] = coarseX[j,i] = X[i*(n-1):(i+1)*(n-1),j*(n-1):(j+1)*(n-1)].sum()
            
    return coarseX

def shuffled_entropy(fim, n_iters, n=50):
    """Shuffle FIM entries and calculate block-averaged entropy.
    
    This serves as null for the pivotal neuron structure.
    
    Parameters
    ----------
    fim : ndarray
    n_iters : int
    n : int, 50 
        Number of spins.
    
    Returns
    -------
    list of ndarray of length n_iters
    """
    
    S = []
    diag = fim.diagonal()
    offdiag = fim[np.triu_indices_from(fim, k=1)]
    
    for i in range(n_iters):
        mixedfim = np.zeros_like(fim)
        mixedfim[np.triu_indices_from(fim, k=1)] = np.random.permutation(offdiag)
        mixedfim += mixedfim.T
        mixedfim[np.diag_indices_from(fim)] = np.random.permutation(diag)
        
        val, vec = sorted_eigh(block_mean(n, mixedfim))

        p = vec**2
        S.append(-(p * np.log2(p)).sum(0))

    return S

def sorted_eigh(X):
    """
    Parameters
    ----------
    X : ndarray

    Returns
    -------
    ndarray
    ndarray
    """

    val, vec = np.linalg.eigh(X)
    sortix = np.argsort(val)[::-1]
    return val[sortix], vec[:,sortix]

def cfim_entropy(fim, n=50, n_iters=100):
    """Calculate entropy of the eigenvectors of block-summed FIMs for each specified
    solution.
    
    Entropy is \sum_i v_i^2 * log_2( v_i^2 ).
    
    Parameters
    ----------
    fim : ndarray
    n : int, 50
        Number of spins.
    n_iters : int, 100
        Number of random samples to iterate over.
        
    Returns
    -------
    ndarray
        Entropy for block-summed FIM.
    ndarray
        Entropy for block-summed, shuffled FIM.
    """
    
    cfim = block_sum(n, fim)
    val, vec = sorted_eigh(cfim)

    p = vec**2
    S = -(p * np.log2(p)).sum(0)
    mixedS = shuffled_entropy(fim, n_iters)
    
    return S, mixedS

def pivotal_by_subspace(name, n_cutoff=50, **soln_kw):
    """Identify pivotal components by principal subspace eigenvalue. Subspace corresponds
    to the diagonal block matrix where the neuron of interest is a matcher.
    
    Parameters
    ----------
    name : tuple 
        For loading specified solution.
    n_cutoff : int, 50
    **soln_kw
        
    Returns
    -------
    ndarray
        Neuron index sorted by subspace eigenvalue.
    ndarray
        Principal component eigenvalue.
    """
    
    from .organizer import CoupSolution
    soln = CoupSolution(*name, **soln_kw)
    fim = soln.fim()
    
    val, vec = block_subspace_eig(fim)
    v = np.array([v_[0] for v_ in val])
    sortix = np.argsort(v)[::-1][:n_cutoff]
    
    return sortix, v[sortix]

def pivotal_by_colnorm(name, threshold, n_modes,
                       measure='sq norm'):
    """Identify pivotal components by principal subspace eigenvalue.
    
    Parameters
    ----------
    name : tuple 
        For loading specified solution using CoupSolution.
    threshold : int
        Column weight threshold calculated from the sum of squares.
    n_modes : int
        Max rank to consider.
    measure : str, 'sq norm'
        'sq norm' for sum of squares, 'uniformity' for uniformity measure
        
    Returns
    -------
    ndarray
        Neuron index.
    ndarray
        Principal component eigenvalue.
    """
    
    from .organizer import CoupSolution
    assert measure in ['sq norm','uniformity']

    soln = CoupSolution(*name)
    eigvec = soln.eig()[1]

    # in the top eigenvectors, what are the most "important" neurons
    neuronix = []
    colweight = np.zeros((n_modes, 50))
    
    if measure=='sq norm':
        for i in range(n_modes):
            mat = vec2mat(eigvec[:,i])
            colweight[i] = (mat**2).sum(0)
    else:
        for i in range(n_modes):
            mat = vec2mat(eigvec[:,i])
            colweight[i] = (mat.sum(0))**2

    # extract those neurons for calculating sensitivity to them specifically
    ix = np.unique(np.concatenate([np.where(colweight[i]>threshold)[0]
                                   for i in range(colweight.shape[0])]))
    
    return ix

def block_subspace_eig(hess, remove_n_modes=0):
    """Spectral analysis of diagonal blocks in the FIM that correspond to individual
    subspaces.

    Parameters
    ----------
    hess : ndarray
    remove_n_modes: int, 0
        Number of modes to subtract off the FIM before extracting the blocks.

    Returns
    -------
    list of ndarray
        Eigenvalue spectrum sorted for each block.
    list of ndarray
        Eigenvectors likewise sorted for each block.
    """
    
    n = (1+np.sqrt(1+4*hess.shape[0])) / 2
    assert int(n)==n, "Cannot be reshaped into n,n pairwise matrix with zeroed diagonal."
    n = int(n)
    k = n-1

    if remove_n_modes>0:
        for i in range(remove_n_modes):
            hess = remove_principal_mode(hess)
    
    # for each spin subspace, perform spectral analysis
    blockeigval = []
    blockeigvec = []
    
    for j in range(len(hess)//k):
        subspaceHess = hess[j*k:(j+1)*k, j*k:(j+1)*k]
        u, v = np.linalg.eigh(subspaceHess)
        sortix = np.argsort(u)[::-1]
        u = u[sortix]
        v = v[:,sortix]

        blockeigval.append(u)
        blockeigvec.append(v)
    return blockeigval, blockeigvec

def subspace_eig(hess, compix):
    """Spectral analysis of subspaces corresponding to indicated components.

    Parameters
    ----------
    hess : ndarray
    ix : list

    Returns
    -------
    list of ndarray
        Eigenvalue spectrum.
    list of ndarray
        Eigenvectors.
    """
    
    n = (1+np.sqrt(1+4*hess.shape[0])) / 2
    assert int(n)==n, "Cannot be reshaped into n,n pairwise matrix with zeroed diagonal."
    n = int(n)
    k = n-1

    # obtain corresponding indices in FIM
    rowix = []
    for ix in compix:
        rowix += list(range(ix*k, (ix+1)*k))
    
    # calculate spectrum and sort by eigenvalue
    subspaceHess = hess[rowix][:,rowix]
    u, v = np.linalg.eigh(subspaceHess)
    sortix = np.argsort(u)[::-1]
    u = u[sortix].real
    v = v[:,sortix].real

    return u, v

def pair_asymmetry(eigvec, rank=0, by_voter=False, eigval=None):
    """Row column asymmetry for "eigenmatrices" for pairwise perturbations. Can calculate
    weighted asymmetry if eigenvalues are passed in.

    Parameters
    ----------
    eigvec : ndarray
        Matrix of eigenvectors where each col is an eigenvector.
    rank : int, 0
        Column of eigvec array to use corresponding to the rank of the eigenvector.
    by_voter : bool, False
        If True, return asymmetry per voter.
    eigval : ndarray, None
        If passed in, a weighted measure of asymmetry will be calculated using the
        relative square of the eigenvalue for each eigenvector.

    Returns
    -------
    float
        Asymmetry measure.
    """
    
    if eigval is None:
        v = vec2mat(eigvec[:,rank])
        if by_voter:
            return ((v-v.T)**2).sum(0)/4
            # old measure (that limited comparison to that of magnitudes)
            #return (v**2+v.T**2-2*np.abs(v*v.T)).sum(0)/2
        return ((v-v.T)**2).sum()/4
        #return (v**2+v.T**2-2*np.abs(v*v.T)).sum()/2
    
    raise NotImplementedError
    # remains to convert the following calculating of asymmetry to the latest definition
    a = 0.
    for rank, val in enumerate(eigval):
        v = vec2mat(eigvec[:,rank])
        if by_voter:
            a += (v**2+v.T**2-2*np.abs(v*v.T)).sum(0)/2 * val**2
        else:
            a += (v**2+v.T**2-2*np.abs(v*v.T)).sum()/2 * val**2
    
    return a/(eigval**2).sum()

def local2neighbor_ratio(eigvec, rank=0):
    """Norm of all localized perturbations to neighborhood perturbations. In other words,
    the column to row ratio.

    Parameters
    ----------
    eigvec : ndarray
        Matrix of eigenvectors where each col is an eigenvector.
    rank : int, 0
        Column of eigvec array to use corresponding to the rank of the eigenvector.
    by_voter : bool, False
        If True, return asymmetry per voter.

    Returns
    -------
    ndarray
        Asymmetry measure per voter.
    """
    
    v = vec2mat(eigvec[:,rank])
    return (v**2).sum(1)/(v**2).sum(0)
    
def norm_entropy(X):
    """Entropy of cols.

    Parameters
    ----------
    X : ndarray
        Square eigenmatrix.
    
    Returns
    -------
    float
    """
    
    assert X.shape[0]==X.shape[1]

    s = (X**2).sum(0)
    assert np.isclose(s.sum(), 1)

    return -np.nansum( s * np.log(s) )

