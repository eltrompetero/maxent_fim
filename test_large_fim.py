# ====================================================================================== #
# Test suite for large_fim.py module.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== # 
from .utils import *
from .large_fim import *
from time import perf_counter
import numdifftools as ndt


def test_Coupling3(n=5, disp=True, time=False):
    """Tests for Coupling3.
    
    This test will take a while because of MC sampling.
    """
    
    from coniii.models import Potts3

    # Setup test
    rng = np.random.RandomState(0)
    h = np.concatenate((rng.normal(size=2*n, scale=.3), np.zeros(n)))
    J = rng.normal(size=n*(n-1)//2, scale=.1)

    # compare stochastic with analytic solution for each pair of perturbations
    # analytic solution finds the exact gradient at the specified parameters
    model = Coupling3(n, h, J, n_samples=10_000_000, eps=1e-4, iprint=False, rng=rng)
    pmodel = Potts3([h,J])

    err = np.zeros(n*(n-1))
    counter = 0
    for i in range(n):
        for j in np.delete(range(n), i):
            sampledGrad = model._solve_linearized_perturbation(i, j)[0]
            trueGrad = model._solve_linearized_perturbation_tester(i, j)
            err[counter] = np.linalg.norm(sampledGrad-trueGrad)
            counter += 1
    err = err.reshape(n,n-1)
    assert err.max()<0.007, err.max()
    
    # compute Hessian directly and compare with (better/more efficient/more precise) code
    # here, a pairwise perturbation from any i to a is possible, leading to n*(n-1)
    # possible perturbations
    pk = model.p2pk(model.p, model.coarseUix, model.coarseInvix)
    log2pk = np.log2(pk)
    def f(eps):
        dJ = model.dJ.T.dot(eps)
        return (log2pk - np.log2(model.p2pk(pmodel.calc_p(model.hJ + dJ),
                                            model.coarseUix,
                                            model.coarseInvix))).dot(pk)
    hessfun = ndt.Hessian(f, step=1e-4)
    hessNdt = hessfun(np.zeros(n*(n-1)))
    if time:
        t0 = perf_counter()
    hessToCheck, errflag, err = model.maj_curvature(epsdJ=1e-7, full_output=True, iprint=False)
    if time:
        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
    
    relErrMax = np.abs((hessNdt-hessToCheck)/hessToCheck).max()
    # some of these derivatives are near 0 so relative error can be large
    relErrTol = 4e-3
    if disp and relErrMax>relErrTol:
        print("Entries in Hessian")
        print(hessNdt[:5,:5]-hessToCheck[:5,:5])
        print()

        print("Relative error")
        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
        print()

        print("NDT error")
        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
        print()
    assert relErrMax<relErrTol, relErrMax
    print("Test passed: Hessian checked with numdifftools.")
