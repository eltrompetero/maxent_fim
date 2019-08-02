# ====================================================================================== #
# Analysis of FIM quantities.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from .utils import *


def block_subspace_eig(hess, k, remove_n_modes=0):
    """Spectral analysis of diagonal blocks in the FIM that correspond to individual
    subspaces.

    Parameters
    ----------
    hess : ndarray
    k : int
        Size of each diagonal block.
    remove_n_modes: int, 0
        Number of modes to subtract off the FIM before extracting the blocks.

    Returns
    -------
    list of ndarray
        Eigenvalue spectrum for each block.
    list of ndarray
        Eigenvectors for each block.
    """
    
    assert (len(hess)%k)==0
    if remove_n_modes>0:
        for i in range(remove_n_modes):
            hess = remove_principal_mode(hess)
    
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

def pair_asymmetry(eigvec, n, rank=0, by_voter=False, eigval=None):
    """Row column asymmetry for "eigenmatrices" for pairwise perturbations. Can calculated
    weighted asymmetry if eigenvalues are passed in.

    Parameters
    ----------
    eigvec : ndarray
        Matrix of eigenvectors where each col is an eigenvector.
    n : int
        System size.
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
        v = np.insert(eigvec[:,rank], range(0,n*n,n), 0).reshape(n,n)
        if by_voter:
            return (v**2+v.T**2-2*np.abs(v*v.T)).sum(0)/2
        return (v**2+v.T**2-2*np.abs(v*v.T)).sum()/2
    
    a = 0.
    for rank, val in enumerate(eigval):
        v = np.insert(eigvec[:,rank], range(0,n*n,n), 0).reshape(n,n)
        if by_voter:
            a += (v**2+v.T**2-2*np.abs(v*v.T)).sum(0)/2 * val**2
        else:
            a += (v**2+v.T**2-2*np.abs(v*v.T)).sum()/2 * val**2
    
    return a/(eigval**2).sum()
