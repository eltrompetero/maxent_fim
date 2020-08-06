# ============================================================================================ #
# Test module for utils.py
# Author : Eddie Lee, edlee@santafe.edu
# ============================================================================================ # 
from .utils import *
from time import perf_counter
np.random.seed(0)



def test_coarse_grain():
    X = np.random.choice([-1,1], size=(100,3))
    coarseX, groupsix = coarse_grain(X, 3)
    assert np.array_equal(X, coarseX)
    
    # with 4 cols and 3 bins, the first two cols should be preserved
    X = np.random.choice([-1,1], size=(100,4))
    coarseX, groupsix = coarse_grain(X, 3)
    assert np.array_equal(coarseX[:,:2], X[:,:2])
    assert ~np.array_equal(coarseX[:,-1], np.sign(X[:,-2:].sum(1)))

def test_tweak_constraints(n=3):
    X=np.random.choice([-1,1],(100,n))
    sisj=pair_corr(X, concat=True)
    for i in range(n):
        for j in np.delete(np.arange(n),i):
            Xdup=X.copy()
            Xdup[:,i]=X[:,j]
            assert np.isclose( pair_corr(Xdup, concat=True),
                               tweak_constraints(sisj, i, j, 1, n) ).all()

def test_remove_principal_mode():
    X = np.corrcoef(np.random.rand(5,5))
    el, v = np.linalg.eig(X)
    sortix = np.argsort(el)[::-1][1:-1]
    el = el[sortix].real
    v = v[:,sortix].real
    
    newX = remove_principal_mode(X)
    elnew, vnew = np.linalg.eig(newX)
    sortix = np.argsort(elnew)[::-1][:-2]
    elnew = elnew[sortix].real
    vnew = vnew[:,sortix].real
    #print(elnew)
    #print(el)
    #print()
    #print(vnew)
    #print(v)
    assert np.isclose(elnew, el).all(), (elnew, el)
    assert np.isclose(np.abs(vnew.T.dot(v)),np.eye(3)).all(), (vnew, v)

def test_enumerate_unique_splits():
    for i in range(3, 22, 3):
        splits = enumerate_unique_splits(i)
        assert (splits.sum(1)==i).all()
        print("Test passed: splits add up correctly.")

        assert (np.diff(splits[:,0])<=0).all()
        print("Test passed: largest group monotonically decreases in specified order.")

        assert (np.diff(splits, axis=1)<=0).all()
        print("Test passed: Preceding groups are always larger than the next.")
