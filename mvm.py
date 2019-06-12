# =============================================================================================== #
# Module for Median Voter Model.
# Author: Eddie Lee, edl56@cornell.edu
# =============================================================================================== #
import numpy as np
from coniii.utils import *


def create_mvm_p(n, q):
    """Use a probability transition matrix formulation to generate the MVM probability
    distribution. This can be quite expensive when generating p.

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

def mvm_corr(n):
    """Median Voter Model (q=1) pairwise correlations.
    
    Parameters
    ----------
    int : n
    
    Returns
    -------
    float
        <s_median * s_ordinary>
    float
        <s_o * s_o''>
    """
    
    assert (n%2)==1
    Z = 2**(n-2) 
    
    # <s_Median s_Ordinary>
    smo = (sum([binom(n-1,k) * (k/(n-1) - (n-1-k)/(n-1))
                for k in range((n-1)//2,n)])) / Z
    
    soo = 0.
    return smo, soo

def couplings(n, full_output=False):
    """Find couplings corresponding to mvm pairwise correlations numerically.

    Parameters
    ----------
    n : int
    full_output : bool, False

    Returns
    -------
    ndarray
        [Jmo, Joo]
    """
    
    from scipy.optimize import minimize

    smo, soo = mvm_corr(n)
    smo_fun, _, soo_fun, _, _ = setup_maxent_mvm(n)
    def cost(params):
        Jmo, Joo = params
        return np.sqrt((smo-smo_fun(Jmo, Jmo, Joo, Joo))**2 +
                       (soo-soo_fun(Jmo, Jmo, Joo, Joo))**2)

    soln = minimize(cost, [0,0])
    if full_output:
        return soln['x'], soln
    return soln['x']

def setup_maxent_mvm(n):
    """Median Voter Model with special Ordinary voter O' that has special couplings with
    the Median and the remaining O voters.
    
    Check formulation in SCOTUS II pg. 116.
    
    Parameters
    ----------
    int : n
    
    Returns
    -------
    function
        smo(Jm, Jmp, Jo, Jop)
    function
        smo_prime(Jm, Jmp, Jo, Jop)
    function
        soo(Jm, Jmp, Jo, Jop)
    function
        soo_prime(Jm, Jmp, Jo, Jop)
    """
    
    _E_with_maj_with_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(2*k-n-2) +
                                          Jmp +
                                          Jo*(binom(k-2,2)+binom(n-k,2)-(k-2)*(n-k)) +
                                          Jop*(2*k-n-2))
    _E_with_maj_against_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(2*k-n) -
                                           Jmp +
                                           Jo*(binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1)) +
                                           Jop*(n-2*k))
    _E_not_with_maj_with_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(n-2*k-2) +
                                          Jmp +
                                          Jo*(binom(n-k-2,2)+binom(k,2)-(n-k-2)*k) +
                                          Jop*(n-2*k-2))
    _E_not_with_maj_against_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(n-2*k) -
                                            Jmp +
                                            Jo*(binom(n-k-1,2)+binom(k-1,2)-(n-k-1)*(k-1)) +
                                            Jop*(2*k-n))
    
    Z = lambda Jm,Jmp,Jo,Jop,n=n:( sum([binom(n,k) * 
           (k/n * ((k-1)/(n-1) * np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) +
                   (n-k)/(n-1) * np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,k))) +
           (n-k)/n * ((n-k-1)/(n-1) * np.exp(-_E_not_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) + 
                      k/(n-1) * np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,k))))
                            for k in range(n//2+1,n-1)]) + 
           n*((n-1)/n * ((n-2)/(n-1) * np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n-1)) +
                   1/(n-1) * np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
                   1/n * np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) + 
           np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n)))
    
    # <s_Median s_Ordinary>
    def smo(Jm, Jmp, Jo, Jop, n=n):
        return (sum([binom(n,k) * 
           (k/n * ((k-1)/(n-1) * ((k-2)/(n-2) - (n-k)/(n-2)) * 
                                   np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) +
                   (n-k)/(n-1) * ((k-1)/(n-2) - (n-k-1)/(n-2)) * 
                                   np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,k))) +
           (n-k)/n * ((n-k-1)/(n-1) * ((n-k-2)/(n-2) - k/(n-2)) * 
                                  np.exp(-_E_not_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) + 
                      k/(n-1) * ((n-k-1)/(n-2) - (k-1)/(n-2)) * 
                                  np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,k))))
                            for k in range(n//2+1,n-1)]) + 
            n*((n-1)/n * ((n-2)/(n-1) * ((n-3)/(n-2) - 1/(n-2)) *
                                  np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n-1)) +
                   1/(n-1) * np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
                   -1/n * np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) + 
            np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n))) / Z(Jm,Jmp,Jo,Jop)
    
    # <s_M s_O'>
    def smop(Jm, Jmp, Jo, Jop, n=n):
        return (sum([binom(n,k) * 
           (k/n * ((k-1)/(n-1) * np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) +
                   (n-k)/(n-1) * -np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,k))) +
           (n-k)/n * ((n-k-1)/(n-1) * np.exp(-_E_not_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) + 
                      k/(n-1) * -np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,k))))
                            for k in range(n//2+1,n-1)]) + 
            n*((n-1)/n * ((n-2)/(n-1) * np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n-1)) +
                   1/(n-1) * -np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
               1/n * ( -np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1)))) + 
            np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n))) / Z(Jm,Jmp,Jo,Jop)
    
    # <s_O s_O''>
    def soo(Jm, Jmp, Jo, Jop, n=n):
        return (sum([binom(n,k) * 
           (k/n * ((k-1)/(n-1) * (binom(k-2,2)+binom(n-k,2)-(k-2)*(n-k))/binom(n-2,2) * 
                                   np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) +
                   (n-k)/(n-1) * (binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1))/binom(n-2,2) * 
                                   np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,k))) +
           (n-k)/n * ((n-k-1)/(n-1) * (binom(n-k-2,2)+binom(k,2)-(n-k-2)*k)/binom(n-2,2) * 
                                  np.exp(-_E_not_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) + 
                      k/(n-1) * (binom(n-k-1,2)+binom(k-1,2)-(n-k-1)*(k-1))/binom(n-2,2) * 
                                  np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,k))))
                            for k in range(n//2+1,n-1)]) + 
           n*((n-1)/n * ((n-2)/(n-1) * (binom(n-3,2)-(n-3))/binom(n-2,2) * 
                                   np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n-1)) +
                   1/(n-1) * np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
               1/n * np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
            np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n))) / Z(Jm,Jmp,Jo,Jop)
    
    # <s_O s_O'>
    def sop(Jm, Jmp, Jo, Jop, n=n):
        return (sum([binom(n,k) * 
       (k/n * ( (k-1)/(n-1) * ((k-2)/(n-2) - (n-k)/(n-2)) * 
                       np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) +
                (n-k)/(n-1) * ((n-k-1)/(n-2) - (k-1)/(n-2)) * 
                       np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,k)) ) +
       (n-k)/n * ( (n-k-1)/(n-1) * ((n-k-2)/(n-2) - k/(n-2)) * 
                       np.exp(-_E_not_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) + 
                k/(n-1) * ((k-1)/(n-2) - (n-k-1)/(n-2)) * 
                       np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,k))))
                            for k in range(n//2+1,n-1)]) + 
       n*((n-1)/n * ( (n-2)/(n-1) * ((n-3)/(n-2) - 1/(n-2)) * 
                       np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n-1)) +
                      -1/(n-1) * np.exp(-_E_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1)) ) +
          1/n * np.exp(-_E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
       np.exp(-_E_with_maj_with_median(Jm,Jmp,Jo,Jop,n))) / Z(Jm,Jmp,Jo,Jop)
    return smo, smop, soo, sop, Z


