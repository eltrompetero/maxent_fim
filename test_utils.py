# ============================================================================================ #
# Quick access to useful modules from SCOTUS project.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ============================================================================================ # 
from .utils import *


def test_tweak_constraints(n=3):
    X=np.random.choice([-1,1],(100,n))
    sisj=pair_corr(X, concat=True)
    for i in range(n):
        for j in np.delete(np.arange(n),i):
            Xdup=X.copy()
            Xdup[:,i]=X[:,j]
            assert np.isclose( pair_corr(Xdup, concat=True),
                               tweak_constraints(sisj, i, j, 1, n) ).all()

def test_IsingFisherCurvatureMethod1():
    import numdifftools as ndt

    n = 5
    rng = np.random.RandomState(0)
    hJ = rng.normal(scale=.1, size=15)
    isingdkl = IsingFisherCurvatureMethod1(n, h=hJ[:n], J=hJ[n:])

    # Compare linearized perturbation matrix calculation with direct solution by solving
    # the inverse problem.
    for i in range(3):
        if isingdkl.sisj[i]>0:
            dJ = isingdkl._solve_linearized_perturbation(i, perturb_up=False)
        else:
            dJ = isingdkl._solve_linearized_perturbation(i)
        assert np.linalg.norm(dJ-isingdkl.dJ[i])<1e-6
    
    # Compare own estimation of Hessian with simpler (but slower) implementation using numdifftools.
    # Checking within 1% error.
    log2p = np.log2(isingdkl.p)
    def f(eps):
        dJ = (eps[:,None]*isingdkl.dJ).sum(0)
        return (log2p-np.log2(isingdkl.ising.p(isingdkl.hJ + dJ))).dot(isingdkl.p)

    hessfun = ndt.Hessian(f, step=1e-5)
    hessNdt = hessfun(np.zeros(n))
    hessToCheck = isingdkl.dkl_curvature()
    assert (np.abs((hessNdt-hessToCheck)/hessToCheck)<1e-2).all()
