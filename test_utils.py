# ============================================================================================ #
# Test module for utils.py
# Author : Eddie Lee, edlee@santafe.edu
# ============================================================================================ # 
from .utils import *
from time import perf_counter



def test_perturb_3_spin():
    np.random.seed(0)
    
    # check symmetric case
    p = np.ones(3) / 3
    for i in range(3):
        delta = perturb_3_spin(p, i, eps=1e-4, return_delta=True)
        assert delta[i]>0 and np.isclose(delta[(i+1)%2], delta[(i-1)%2]) and delta[i]!=delta[(i-1)%2]
        
        delta = perturb_3_spin(p, i, eps=-1e-4, return_delta=True)
        assert delta[i]<0 and np.isclose(delta[(i+1)%2], delta[(i-1)%2]) and delta[i]!=delta[(i-1)%2]

    for i in range(100):
        p = np.random.rand(3)
        p /= p.sum()
        
        for j in range(3):
            perturb_3_spin(p, j, run_checks=True)

    for i in range(100):
        p = np.random.rand(3)
        p /= p.sum()
        
        for j in range(3):
            perturb_3_spin(p, j, run_checks=True, eps=-1e-4)

def test_coarse_grain():
    np.random.seed(0)
    X = np.random.choice([-1,1], size=(100,3))
    coarseX, groupsix = coarse_grain(X, 3)
    assert np.array_equal(X, coarseX)
    
    # with 4 cols and 3 bins, the first two cols should be preserved
    X = np.random.choice([-1,1], size=(100,4))
    coarseX, groupsix = coarse_grain(X, 3)
    assert np.array_equal(coarseX[:,:2], X[:,:2])
    assert ~np.array_equal(coarseX[:,-1], np.sign(X[:,-2:].sum(1)))

def test_remove_principal_mode():
    np.random.seed(0)
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
