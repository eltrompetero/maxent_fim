# =============================================================================================== #
# Calculation of FIM quantities.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
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
    list
        Eigenvalue spectrum for each block.
    list
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
