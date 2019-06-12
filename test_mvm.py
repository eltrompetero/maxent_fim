# =============================================================================================== #
# Test module for Median Voter Model.
# Author: Eddie Lee, edl56@cornell.edu
# =============================================================================================== #
from .mvm import *
from importlib import import_module


def test_setup_maxent_mvm():
    np.random.seed(0)
    Jmo, Jmop, Joo, Jop = np.random.normal(size=4, scale=.3)
    nRange = [5,7,9,11,13]

    for i,n in enumerate(nRange):
        ising = import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        hJ = np.zeros(n+n*(n-1)//2)
        hJ[n:2*n-1] = Jmo
        hJ[n] = Jmop
        hJ[2*n-1:] = Joo
        hJ[2*n-1:2*n-1+n-2] = Jop
        sisjME = ising.calc_observables(hJ)
        # extract corresponding pairwise correlations from full pairwise maxent model
        smoME, smopME, sooME, sopME = sisjME[n+1], sisjME[n], sisjME[-1], sisjME[n+n-1]
         
        smo, smop, soo, sop, Z = setup_maxent_mvm(n)
        assert np.isclose( smoME, smo(Jmo, Jmop, Joo, Jop) )
        assert np.isclose( smopME, smop(Jmo, Jmop, Joo, Jop) )
        
        assert np.isclose( sooME, soo(Jmo, Jmop, Joo, Jop) )
        assert np.isclose( sopME, sop(Jmo, Jmop, Joo, Jop) )
    print("Test passed: Pairwise correlations agree with ConIII module.")
