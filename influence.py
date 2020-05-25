# ====================================================================================== #
# Analysis of features of FIM that could correspond to measures of "influence."
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *



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
        Eigenvalue spectrum for each block.
    list of ndarray
        Eigenvectors for each block.
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
        u, v = np.linalg.eig(subspaceHess)
        sortix = np.argsort(u)[::-1]
        u = u[sortix]
        v = v[:,sortix]

        blockeigval.append(u)
        blockeigvec.append(v)
    return blockeigval, blockeigvec

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
