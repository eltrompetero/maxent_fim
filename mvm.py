# =============================================================================================== #
# Module for Median Voter Model.
# Author: Eddie Lee, edl56@cornell.edu
# =============================================================================================== #
import numpy as np
from coniii.utils import *


def create_mvm_p(n, q):
    """Use a probability transition matrix formulation to generate the MVM probability distribution.

    Parameters
    ----------
    n : int
    q : float

    Returns
    -------
    ndarray
        Of length 2^n
    """

    allStates = bin_states(n, True)
    T = np.zeros((2**n,2**n))
    for i,s in enumerate(allStates):
        if s[1:].sum()!=0 and s[0]!=np.sign(s[1:].sum()):
            T[i,i] = 1-q
            snew = s.copy()
            snew[0] *= -1
            ix = np.where((allStates==snew).all(1))[0]
            T[ix,i] = q
        else:
            T[i,i] = 1

    p = np.ones(2**n)/2**n
    pmvm = T.dot(p)
    
    if q==1:
        # check that states with zero probability are now ones where the Median was in the minority
        ix = allStates[pmvm>0][:,0]==np.sign(allStates[pmvm>0][:,1:].sum(1))
        assert (np.sign(allStates[pmvm>0][ix==0][:,1:].sum(1))==0).all()
    return pmvm
