# ====================================================================================== #
# Test module for influence.py
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from .influence import *


def test_block_subspace_eig(n=10, rng=np.random.RandomState(0)):
    hess = np.zeros((n,n))
    hess[np.diag_indices(n)] = rng.rand(n)
    eigval, eigvec = block_subspace_eig(hess, 1)
    assert all([np.array_equal([hess[i,i]], v) for i,v in enumerate(eigval)])
    print("Test passed: Diagonal elements are correctly extracted for diagonal matrix.")

def test_pair_asymmetry(n=10, rng=np.random.RandomState(0), n_samples=100):
    for i in range(n_samples):
        x = np.random.rand(n*(n-1))
        x /= np.linalg.norm(x)
        a = pair_asymmetry(x[:,None],n)
        assert np.all((a>=0)&(a<=1)) and 0<=a.sum()<=1
    print("Test passed: asymmetry measure is properly normalized.")
