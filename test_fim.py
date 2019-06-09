# ====================================================================================== #
# Test suite for fim.py module.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== # 
from .utils import *
from .fim import *
from time import perf_counter
import numdifftools as ndt
np.random.seed(0)


def test_IsingFisherCurvatureMethod1(disp=True, time=False):
    n = 5
    rng = np.random.RandomState(0)
    hJ = rng.normal(scale=.1, size=15)
    isingdkl = IsingFisherCurvatureMethod1(n, h=hJ[:n], J=hJ[n:])

    # Compare linearized perturbation matrix calculation with direct solution by solving
    # the inverse problem.
    for i in range(3):
        dJ = isingdkl._solve_linearized_perturbation_tester(i)
        assert np.linalg.norm(dJ-isingdkl.dJ[i])<1e-6, (dJ,isingdkl.dJ[i])
    
    # p(k) in maj 
    pk = isingdkl.p2pk(isingdkl.p, isingdkl.coarseUix, isingdkl.coarseInvix)
    log2pk = np.log2(pk)
    def f(eps):
        dJ = (eps[:,None]*isingdkl.dJ).sum(0)
        return (log2pk - np.log2(isingdkl.p2pk(isingdkl.ising.p(isingdkl.hJ + dJ),
                                               isingdkl.coarseUix,
                                               isingdkl.coarseInvix))).dot(pk)

    hessfun = ndt.Hessian(f, step=1e-4)
    hessNdt = hessfun(np.zeros(n))
    if time:
        t0 = perf_counter()
    hessToCheck, errflag, err = isingdkl.maj_curvature(epsdJ=1e-7, full_output=True)
    if time:
        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
    
    if disp:
        print(hessNdt[:3,:3])
        print(hessToCheck[:3,:3])

        print("Relative error")
        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
        print()

        print("NDT error")
        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
    assert (np.abs((hessNdt-hessToCheck)/hessToCheck)<1e-5).all()

def test_IsingFisherCurvatureMethod1a(disp=True, time=False):
    n = 5
    rng = np.random.RandomState(0)
    hJ = rng.normal(scale=.1, size=15)
    isingdkl = IsingFisherCurvatureMethod1a(n, h=hJ[:n], J=hJ[n:])

    # Compare linearized perturbation matrix calculation with direct solution by solving
    # the inverse problem.
    for i in range(3):
        dJ = isingdkl._solve_linearized_perturbation_tester(i)
        assert np.linalg.norm(dJ-isingdkl.dJ[i])<1e-6, (dJ,isingdkl.dJ[i])
    
    # p(k) in maj 
    pk = isingdkl.p2pk(isingdkl.p, isingdkl.coarseUix, isingdkl.coarseInvix)
    log2pk = np.log2(pk)
    def f(eps):
        dJ = (eps[:,None]*isingdkl.dJ).sum(0)
        return (log2pk - np.log2(isingdkl.p2pk(isingdkl.ising.p(isingdkl.hJ + dJ),
                                               isingdkl.coarseUix,
                                               isingdkl.coarseInvix))).dot(pk)

    hessfun = ndt.Hessian(f, step=1e-4)
    hessNdt = hessfun(np.zeros(n))
    if time:
        t0 = perf_counter()
    hessToCheck, errflag, err = isingdkl.maj_curvature(epsdJ=1e-7, full_output=True)
    if time:
        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
    
    if disp:
        print(hessNdt[:3,:3])
        print(hessToCheck[:3,:3])

        print("Relative error")
        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
        print()

        print("NDT error")
        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
    assert (np.abs((hessNdt-hessToCheck)/hessToCheck)<1e-5).all()

def test_IsingFisherCurvatureMethod2(n=5, disp=True, time=False):
    rng = np.random.RandomState(0)
    hJ = rng.normal(scale=.1, size=n*(n-1)//2+n)
    isingdkl = IsingFisherCurvatureMethod2(n, h=hJ[:n], J=hJ[n:])

    # Compare linearized perturbation matrix calculation with direct solution by solving
    # the inverse problem.
    i = 0
    for a in range(1,4):
        dJ = isingdkl._solve_linearized_perturbation_tester(i, a)
        assert np.linalg.norm(dJ-isingdkl.dJ[a-1])<1e-4, np.linalg.norm(dJ-isingdkl.dJ[a-1])
    
    # p(k) in maj 
    pk = isingdkl.p2pk(isingdkl.p, isingdkl.coarseUix, isingdkl.coarseInvix)
    log2pk = np.log2(pk)
    def f(eps):
        dJ = (eps[:,None]*isingdkl.dJ).sum(0)
        return (log2pk - np.log2(isingdkl.p2pk(isingdkl.ising.p(isingdkl.hJ + dJ),
                                               isingdkl.coarseUix,
                                               isingdkl.coarseInvix))).dot(pk)

    hessfun = ndt.Hessian(f, step=1e-4)
    hessNdt = hessfun(np.zeros(n*(n-1)))
    if time:
        t0 = perf_counter()
    hessToCheck, errflag, err = isingdkl.maj_curvature(epsdJ=1e-7, full_output=True)
    if time:
        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
    
    if disp:
        print(hessNdt[:3,:3])
        print(hessToCheck[:3,:3])

        print("Relative error")
        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
        print()

        print("NDT error")
        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
    assert (np.abs((hessNdt-hessToCheck)/hessToCheck)<1e-6).all()

def test_IsingSpinReplacementFIM(n=4, disp=True, time=False):
    rng = np.random.RandomState(0)
    hJ = rng.normal(scale=.1, size=n*(n-1)//2+n)
    isingdkl = IsingSpinReplacementFIM(n, h=hJ[:n], J=hJ[n:], eps=1e-2)

    for i,j in product(range(n), range(n)):
        if i!=j:
            T = isingdkl.pair_transition_matrix(i, j)
            assert np.isclose(T.sum(0),1).all()
    if disp:
        print("Test passed: All cols of transition matrix sum to 1.")

#def test_IsingFisherCurvatureMethod3():
#    n = 5
#    rng = np.random.RandomState(0)
#    hJ = rng.normal(scale=.1, size=15)
#    isingdkl3 = IsingFisherCurvatureMethod3(n, h=hJ[:n], J=hJ[n:])
#    isingdkl2 = IsingFisherCurvatureMethod2(n, h=hJ[:n], J=hJ[n:])
#    isingdkl1 = IsingFisherCurvatureMethod1(n, h=hJ[:n], J=hJ[n:])
#    
#    # use the other two methods to check this one which bundles calculations from them
#    assert np.isclose(isingdkl3.dJ[:n],isingdkl1.dJ).all()
#    assert np.isclose(isingdkl3.dJ[n:],isingdkl2.dJ).all(), (isingdkl3.dJ[n],isingdkl2.dJ[0])
#
#def test_IsingFisherCurvatureMethod4(n=5, disp=True, time=False):
#    rng = np.random.RandomState(0)
#    k = 3
#    hJ = rng.normal(scale=.1, size=n*(n-1)//2+k*n)
#    isingdkl = IsingFisherCurvatureMethod4(n, k, h=hJ[:n*k], J=hJ[k*n:])
#
#    # Compare linearized perturbation matrix calculation with direct solution by solving
#    # the inverse problem.
#    i = 0
#    for a in range(1,4):
#        dJ = isingdkl._solve_linearized_perturbation_tester(i, a)[0]
#        assert np.linalg.norm(dJ-isingdkl.dJ[a-1])<1e-6
#    
#    # Compare own estimation of Hessian with simpler (but slower) implementation using numdifftools.
#    # p(k) in maj 
#    pk = isingdkl.p2pk(isingdkl.p, isingdkl.coarseUix, isingdkl.coarseInvix)
#    log2pk = np.log2(pk)
#    def f(eps):
#        dJ = (eps[:,None]*isingdkl.dJ).sum(0)
#        return (log2pk - np.log2(isingdkl.p2pk(isingdkl.ising.p(isingdkl.hJ + dJ),
#                                               isingdkl.coarseUix,
#                                               isingdkl.coarseInvix))).dot(pk)
#
#    hessfun = ndt.Hessian(f, step=1e-5)
#    hessNdt = hessfun(np.zeros(n*(n-1)))
#    if time:
#        t0 = perf_counter()
#    hessToCheck, errflag, err = isingdkl.maj_curvature(epsdJ=1e-5, full_output=True)
#    if time:
#        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
#    
#    if disp:
#        print("Relative error")
#        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
#        print()
#
#        print("NDT error")
#        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
#    assert (np.abs((hessNdt-hessToCheck)/hessToCheck)<1e-5).all(), (
#            np.abs((hessNdt-hessToCheck)/hessToCheck).max())

def test_IsingFisherCurvatureMethod4a(n=3, disp=True, time=False):
    from coniii.utils import convert_params
    rng = np.random.RandomState(0)
    k = 3

    # Test with field = 0 ============================================================= #
    hJ = rng.normal(scale=.5, size=n*(n-1)//2+k*n)
    hJ[:-n] = 0
    hJ[:n*k] -= np.tile(hJ[:n],k)

    isingdkl = IsingFisherCurvatureMethod4a(n, k, h=hJ[:n*k], J=hJ[k*n:])
    for ijix,(i,j) in enumerate(combinations(range(n),2)):
        tot = 0
        for gamma in range(k):
            ix = isingdkl.pairs[(gamma,i,gamma,j)]
            tot += isingdkl.p.dot(ix)
        assert np.isclose(isingdkl.sisj[ijix+n*k],tot), (isingdkl.sisj[ijix+n*k],tot)
    print("Test passed: pairwise correlation is sum of probabilities of agreement for each k.")
    
    for i in range(3):
        for k_ in range(k):
            sisj = isingdkl.observables_after_perturbation(i, k_)[0]
            assert np.isclose(sum([sisj[i+k*ix] for ix in range(k)]), 1)
            for j in range(n):
                if j!=i:
                    if i<j:
                        ijix = unravel_index((i,j),n)
                    else:
                        ijix = unravel_index((j,i),n)
                    assert sisj[k*n+ijix]!=isingdkl.sisj[k*n+ijix]
            # check that other pairwise correlations (e.g. for i+1) are not perturbed
            i_ = (i+1)%n
            for j in range(n):
                if j!=i and j!=i_:
                    if i_<j:
                        ijix = unravel_index((i_,j),n)
                    else:
                        ijix = unravel_index((j,i_),n)
                    assert sisj[k*n+ijix]==isingdkl.sisj[k*n+ijix], (i_,j)
    print("Test passed: only focus pairs are perturbed (observables).")
    
    eps = 1e-4
    for gamma in range(3):
        for i in range(3):
            sisj = isingdkl.ising.calc_observables(isingdkl.hJ+isingdkl.dJ[i+gamma*n]*eps)
            dsisj = sisj-isingdkl.sisj

            # check that the means changed for the spin we're changing
            if gamma==0:
                assert dsisj[i]>0 and dsisj[i+n]<0 and dsisj[i+2*n]<0
            elif gamma==1:
                assert dsisj[i+n]>0 and dsisj[i]<0 and dsisj[i+2*n]<0
            else:
                assert dsisj[i+2*n]>0 and dsisj[i]<0 and dsisj[i+n]<0
            
            # check that the means didn't change for the other spins
            for j in range(k*n):
                if (j%n)!=i:
                    assert np.isclose(dsisj[j],0)
            
            # check that pairwise correlations for this spin with others changed
            for j in range(n):
                if j!=i:
                    if i<j:
                        ijix = unravel_index((i,j),n)
                    else:
                        ijix = unravel_index((j,i),n)
                    assert not np.isclose(dsisj[n*k+ijix], 0), dsisj[n*k+ijix]
            # check that other pairwise correlations (e.g. for i+1) are not perturbed
            i_ = (i+1)%n
            for j in range(n):
                if j!=i and j!=i_:
                    if i_<j:
                        ijix = unravel_index((i_,j),n)
                    else:
                        ijix = unravel_index((j,i_),n)
                    assert np.isclose(dsisj[k*n+ijix], 0)
    print("Test passed: only focus means are perturbed (dJ).")
    print("Test passed: only focus pairs are perturbed (dJ).")

    i = 0
    for k_ in range(k):
        dJ = isingdkl._solve_linearized_perturbation_tester(i, k_)
        assert np.linalg.norm(dJ-isingdkl.dJ[i+n*k_])<1e-3, (dJ,isingdkl.dJ[i+n*k_])
    print("Test passed: parameter perturbation agrees with direct solution using inverse maxent.")

    pk = isingdkl.p2pk(isingdkl.p, isingdkl.coarseUix, isingdkl.coarseInvix)
    log2pk = np.log2(pk)
    def f(eps):
        dJ = isingdkl.dJ.T.dot(eps)
        return (log2pk - np.log2(isingdkl.p2pk(isingdkl.ising.p(isingdkl.hJ + dJ),
                                               isingdkl.coarseUix,
                                               isingdkl.coarseInvix))).dot(pk)
    hessfun = ndt.Hessian(f, step=1e-4)
    hessNdt = hessfun(np.zeros(n*k))
    if time:
        t0 = perf_counter()
    hessToCheck, errflag, err = isingdkl.maj_curvature(epsdJ=1e-7, full_output=True, iprint=False)
    if time:
        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
    
    relErrMax = np.abs((hessNdt-hessToCheck)/hessToCheck).max()
    # some of these derivatives are near 0 so relative error can be large
    relErrTol = 2e-3
    if disp and relErrMax>relErrTol:
        print(hessNdt[:5,:5]-hessToCheck[:5,:5])

        print("Relative error")
        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
        print()

        print("NDT error")
        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
    assert relErrMax<relErrTol, relErrMax
    print("Test passed: Hessian checked with numdifftools.")

    # Nonzero fields ================================================================== #
    hJ = rng.normal(scale=.5, size=n*(n-1)//2+k*n)
    hJ[:n*k] -= np.tile(hJ[:n],k)
    isingdkl = IsingFisherCurvatureMethod4a(n, k, h=hJ[:n*k], J=hJ[k*n:], n_cpus=1)

    i = 0
    for k_ in range(3):
        dJ = isingdkl._solve_linearized_perturbation_tester(i, k_)
        assert np.linalg.norm(dJ-isingdkl.dJ[i+n*k_])<1e-3, (dJ,isingdkl.dJ[i+n*k_])
    print("Test passed: parameter perturbation agrees with direct solution using inverse maxent.")

    pk = isingdkl.p2pk(isingdkl.p, isingdkl.coarseUix, isingdkl.coarseInvix)
    log2pk = np.log2(pk)
    def f(eps):
        dJ = isingdkl.dJ.T.dot(eps)
        return (log2pk - np.log2(isingdkl.p2pk(isingdkl.ising.p(isingdkl.hJ + dJ),
                                               isingdkl.coarseUix,
                                               isingdkl.coarseInvix))).dot(pk)

    hessfun = ndt.Hessian(f, step=1e-4)
    hessNdt = hessfun(np.zeros(n*k))
    if time:
        t0 = perf_counter()
    hessToCheck, errflag, err = isingdkl.maj_curvature(epsdJ=1e-7, full_output=True, iprint=False)
    if time:
        print("FI for p(k) took %fs to calculate."%(perf_counter()-t0))
    
    relErrMax = np.abs((hessNdt-hessToCheck)/hessToCheck).max()
    relErrTol = 1e-3
    if disp and relErrMax>relErrTol:
        print("Hess ndt")
        print(hessNdt[:5,:5])
        print('Hess linear')
        print(hessToCheck[:5,:5])
        print('Hess diff')
        print(hessNdt[:5,:5]-hessToCheck[:5,:5])

        print("Relative error")
        print(np.linalg.norm(err)/np.linalg.norm(hessToCheck))
        print()

        print("NDT error")
        print(np.sort(np.abs((hessNdt-hessToCheck)/hessToCheck).ravel())[::-1][:20])
    assert relErrMax<relErrTol, relErrMax
    print("Test passed: Hessian checked with numdifftools.")

def test_perturbations():
    """Check perturbations with random samples."""
    
    n = 4
    eps = .01

    X = np.random.choice([-1,1], size=(10_000_000,n))
    si, sisj = pair_corr(X)
    jit_observables_after_perturbation_plus(n, si, sisj, 0, 1, eps)
    
    ix = np.random.rand(len(X))<eps
    X[ix,0] = X[ix,1]
    siSamp, sisjSamp = pair_corr(X)
    assert (np.abs(siSamp-si)<1e-4).all() and (np.abs(sisjSamp-sisj)<1e-4).all(), (siSamp-si, sisjSamp-sisj)
    print("Test passed: jit_observables_after_perturbation_plus compares well with random sample.")
    
    si, sisj = pair_corr(X)
    jit_observables_after_perturbation_plus_mean(n, si, sisj, 0, eps)
    
    ix = (np.random.rand(len(X))<eps) & (X[:,0]==-1)
    X[ix,0] = 1
    siSamp, sisjSamp = pair_corr(X)
    assert (np.abs(siSamp-si)<1e-4).all() and (np.abs(sisjSamp-sisj)<1e-4).all(), (siSamp-si, sisjSamp-sisj)
    print("Test passed: jit_observables_after_perturbation_plus_mean compares well with random sample.")



# ================ #
# Useful functions #
# ================ #
def generate_test_params(n, k=2, seed=0):
    """Generate parameters for putting into a Method class."""
    
    rng = np.random.RandomState(seed)

    if k==2:
        return (n, np.concatenate((np.zeros(n),rng.normal(size=(k-1)*n))),
                rng.normal(size=n*(n-1)//2))
    elif k==3:
        return (n, k, np.concatenate((np.zeros(n),rng.normal(size=(k-1)*n))),
                rng.normal(size=n*(n-1)//2))
    raise NotImplementedError
