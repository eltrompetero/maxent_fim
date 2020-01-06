# ====================================================================================== #
# Module for Median Voter Model.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from coniii.utils import *
from coniii.enumerate import fast_logsumexp
from scipy.special import binom, factorial, comb
from scipy.optimize import minimize, root


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

def corr(n):
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

def couplings(n,
              data_corr=None,
              full_output=False,
              tol=1e-12,
              return_as_full_vec=False):
    """Find couplings corresponding to mvm pairwise correlations numerically. First, a
    solution is found using scipy.minimize. Then an iterative, fixed point algorithm is
    used to refine the solution.

    Parameters
    ----------
    n : int
    data_corr : ndarray, None
        Correlations to fit instead of taking MVM correlations.
        (smo, smop, soo, soop)
    full_output : bool, False
        If True, return output from scipy.minimize.  
    tol : float, 1e-12
        Norm error allowed in fit to pairwise correlations.
    return_as_full_vec : bool, False
        If True, return the couplings J as part of a full (h, J) vector that can be passed
        directly in to the coniii module.

    Returns
    -------
    ndarray
        [Jmo, Joo] or (h, J) vector that can be passed to coniii.
    dict (optional)
        From scipy.minimize.
    """
    
    if data_corr is None:
        smo, soo = corr(n)
        smo_fun, _, soo_fun, _, _ = setup_maxent(n)
        def cost(params, as_vec=False):
            Jmo, Joo = params
            if as_vec:
                return np.array([smo-smo_fun(Jmo, Jmo, Joo, Joo),
                                 soo-soo_fun(Jmo, Jmo, Joo, Joo)])
            return np.sqrt((smo-smo_fun(Jmo, Jmo, Joo, Joo))**2 +
                           (soo-soo_fun(Jmo, Jmo, Joo, Joo))**2)
        soln = minimize(cost, [0,0], tol=1e-10)
        soln = root(lambda x:cost(x, as_vec=True), soln['x'])
    else:
        smo, smop, soo, soop = data_corr
        smo_fun, smop_fun, soo_fun, soop_fun, _ = setup_maxent(n)
        def cost(params, as_vec=False):
            if as_vec:
                return np.array([smo - smo_fun(*params),
                                 smop - smop_fun(*params),
                                 soo - soo_fun(*params),
                                 soop - soop_fun(*params)])
            return np.sqrt((smo - smo_fun(*params))**2 +
                           (smop - smop_fun(*params))**2 + 
                           (soo - soo_fun(*params))**2 +
                           (soop - soop_fun(*params))**2)
        soln = minimize(cost, [.1,.1,0,0], tol=1e-10)
        soln = root(lambda x:cost(x, as_vec=True), soln['x'])

    if return_as_full_vec:
        params = np.zeros(n+n*(n-1)//2)
        params[n:2*n-1] = soln['x'][0]
        params[2*n-1:] = soln['x'][1]
    else:
        params = soln['x']

    if full_output:
        return params, soln
    return params

def setup_fast_mvm(n):
    """Straightforward MVM with only special correlations between the Median 
    and all uniform Ordinary voters.
    
    Check formulation in SCOTUS II pg. 116.
    
    Parameters
    ----------
    int : n
    
    Returns
    -------
    function
        smo(Jmo, Joo)
    function
        soo(Jmo, Joo)
    """
    
    _E_with_maj = lambda Jmo,Joo,k,n=n: -Jmo*(2*k-n-1) - Joo*(binom(k-1,2) + binom(n-k,2) - (k-1)*(n-k))
    _E_not_with_maj = lambda Jmo,Joo,k,n=n: -Jmo*(n-2*k-1) - Joo*(binom(n-k-1,2) + binom(k,2) - (n-k-1)*k)
    
    Z = lambda Jmo,Joo,n=n: sum([k/n * binom(n,k) * np.exp(-_E_with_maj(Jmo,Joo,k)) +
                                 (n-k)/n * binom(n,k) * np.exp(-_E_not_with_maj(Jmo,Joo,k))
                            for k in range(n//2+1,n)]) + np.exp(-_E_with_maj(Jmo,Joo,n))
    # <s_m s_o>
    def smo(Jmo, Joo, n=n):
        return (sum([binom(n,k) * (k/n * (2*k-n-1)/(n-1) * np.exp(-_E_with_maj(Jmo,Joo,k))
                                 +(n-k)/n * (n-2*k-1)/(n-1) * np.exp(-_E_not_with_maj(Jmo,Joo,k)))
                     for k in range(n//2+1,n)]) + np.exp(-_E_with_maj(Jmo,Joo,n)))/Z(Jmo,Joo)
    # <s_o s_o'>
    def soo(Jmo, Joo, n=n):
        weightmaj = lambda k:k/n * (binom(k-1,2)+binom(n-k,2)-(k-1)*(n-k))/binom(n-1,2)
        weightnotmaj = lambda k:(n-k)/n * (binom(k,2)+binom(n-k-1,2)-k*(n-k-1))/binom(n-1,2)
        return (sum([binom(n,k) * (weightmaj(k) * np.exp(-_E_with_maj(Jmo,Joo,k)) + 
                                   weightnotmaj(k) * np.exp(-_E_not_with_maj(Jmo,Joo,k)))
                     for k in range(n//2+1,n)]) + np.exp(-_E_with_maj(Jmo,Joo,n)))/Z(Jmo,Joo)
    return smo, soo, Z

def setup_maxent(n):
    """Correlation functions of the Median Voter Model with special Ordinary voter O' that
    has special couplings with the Median and the remaining O voters. Using more stable
    formulation of logsumexp.
    
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
    function
        pk(Jm, Jmp, Jo, Jop)
        Distribution of k votes in the majority.
    """
    
    E_with_maj_with_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(2*k-n-2) +
                                          Jmp +
                                          Jo*(binom(k-2,2)+binom(n-k,2)-(k-2)*(n-k)) +
                                          Jop*(2*k-n-2))
    E_with_maj_against_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(2*k-n) -
                                          Jmp +
                                          Jo*(binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1)) +
                                          Jop*(n-2*k))
    E_not_with_maj_with_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(n-2*k-2) +
                                          Jmp +
                                          Jo*(binom(n-k-2,2)+binom(k,2)-(n-k-2)*k) +
                                          Jop*(n-2*k-2))
    E_not_with_maj_against_median = lambda Jm,Jmp,Jo,Jop,k,n=n: -(Jm*(n-2*k) -
                                          Jmp +
                                          Jo*(binom(n-k-1,2)+binom(k-1,2)-(n-k-1)*(k-1)) +
                                          Jop*(2*k-n))
   
    def logZ(*J, n=n):
        coeffs = []
        exp = []
        for k in range(n//2+1, n-1):
            coeffs += [k * (k-1) / (n * (n-1)),
                       k * (n-k) / (n * (n-1)),
                       (n-k) * (n-k-1) / (n * (n-1)),
                       (n-k) * k / (n * (n-1))]
            exp += [-E_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_with_maj_against_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_against_median(*J, k) + np.log(binom(n,k))]
        coeffs += [(n-2) / n, 1/n, 1/n, 1]
        exp += [-E_with_maj_with_median(*J, n-1) + np.log(n),
                -E_with_maj_against_median(*J, n-1) + np.log(n),
                -E_not_with_maj_against_median(*J, n-1) + np.log(n),
                -E_with_maj_with_median(*J, n)]
        return fast_logsumexp(exp, coeffs)[0]

    #Z = lambda Jm,Jmp,Jo,Jop,n=n:( sum([binom(n,k) * 
    #       (k/n * ((k-1)/(n-1) * np.exp(-E_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) +
    #               (n-k)/(n-1) * np.exp(-E_with_maj_against_median(Jm,Jmp,Jo,Jop,k))) +
    #       (n-k)/n * ((n-k-1)/(n-1) * np.exp(-E_not_with_maj_with_median(Jm,Jmp,Jo,Jop,k)) + 
    #                  k/(n-1) * np.exp(-E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,k))))
    #                        for k in range(n//2+1,n-1)]) + 
    #       n*((n-1)/n * ((n-2)/(n-1) * np.exp(-E_with_maj_with_median(Jm,Jmp,Jo,Jop,n-1)) +
    #               1/(n-1) * np.exp(-E_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) +
    #               1/n * np.exp(-E_not_with_maj_against_median(Jm,Jmp,Jo,Jop,n-1))) + 
    #       np.exp(-E_with_maj_with_median(Jm,Jmp,Jo,Jop,n)))
    
    # <s_Median s_Ordinary>
    def smo(*J, n=n):
        coeffs = []
        exp = []
        for k in range(n//2+1, n-1):
            coeffs += [k * (k-1) * (2*k-n-2) / (n * (n-1) * (n-2)),
                       k * (n-k) * (2*k-n) / (n * (n-1) * (n-2)),
                       (n-k) * (n-k-1) * (n-2*k-2) / (n * (n-1) * (n-2)),
                       (n-k) * k * (n-2*k) / (n * (n-1) * (n-2))]
            exp += [-E_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_with_maj_against_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_against_median(*J, k) + np.log(binom(n,k))]
        coeffs += [(n-4)/n, 1/n, -1/n, 1]
        exp += [-E_with_maj_with_median(*J, n-1) + np.log(n),
                -E_with_maj_against_median(*J, n-1) + np.log(n),
                -E_not_with_maj_against_median(*J, n-1) + np.log(n),
                -E_with_maj_with_median(*J, n)]
        num, sign = fast_logsumexp(exp, coeffs)
        return sign * np.exp( num - logZ(*J) )

    # <s_M s_O'>
    def smop(*J, n=n):
        coeffs = []
        exp = []
        for k in range(n//2+1, n-1):
            coeffs += [k * (k-1) / (n * (n-1)),
                       k * (k-n) / (n * (n-1)),
                       (n-k) * (n-k-1) / (n * (n-1)),
                       (n-k) * -k / (n * (n-1))]
            exp += [-E_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_with_maj_against_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_against_median(*J, k) + np.log(binom(n,k))]
        coeffs += [(n-2)/n, -1/n, -1/n, 1]
        exp += [-E_with_maj_with_median(*J, n-1) + np.log(n),
                -E_with_maj_against_median(*J, n-1) + np.log(n),
                -E_not_with_maj_against_median(*J, n-1) + np.log(n),
                -E_with_maj_with_median(*J, n)]
        num, sign = fast_logsumexp(exp, coeffs)
        return sign * np.exp( num - logZ(*J) )
    
    # <s_O s_O''>
    def soo(*J, n=n):
        coeffs = []
        exp = []
        for k in range(n//2+1, n-1):
            coeffs += [k * (k-1) * (binom(k-2,2)+binom(n-k,2)-(k-2)*(n-k)) / (n * (n-1) * binom(n-2,2)),
                       k * (n-k) * (binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1)) / (n * (n-1) * binom(n-2,2)),
                       (n-k) * (n-k-1) * (binom(n-k-2,2)+binom(k,2)-(n-k-2)*k) / (n * (n-1) * binom(n-2,2)),
                       (n-k) * k * (binom(n-k-1,2)+binom(k-1,2)-(n-k-1)*(k-1)) / (n * (n-1) * binom(n-2,2))]
            exp += [-E_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_with_maj_against_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_against_median(*J, k) + np.log(binom(n,k))]
        coeffs += [(n-2) * (binom(n-3,2)-(n-3)) / (n * binom(n-2,2)), 1/n, 1/n, 1]
        exp += [-E_with_maj_with_median(*J, n-1) + np.log(n),
                -E_with_maj_against_median(*J, n-1) + np.log(n),
                -E_not_with_maj_against_median(*J, n-1) + np.log(n),
                -E_with_maj_with_median(*J, n)]
        num, sign = fast_logsumexp(exp, coeffs)
        return sign * np.exp( num - logZ(*J) )
 
    # <s_O s_O'>
    def sop(*J, n=n):
        coeffs = []
        exp = []
        for k in range(n//2+1, n-1):
            coeffs += [k * (k-1) * (2*k-n-2) / (n * (n-1) * (n-2)),
                       k * (n-k) * (n-2*k) / (n * (n-1) * (n-2)),
                       (n-k) * (n-k-1) * (n-2*k-2) / (n * (n-1) * (n-2)),
                       (n-k) * k * (2*k-n) / (n * (n-1) * (n-2))]
            exp += [-E_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_with_maj_against_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                    -E_not_with_maj_against_median(*J, k) + np.log(binom(n,k))]
        coeffs += [(n-4)/n, -1/n, 1/n, 1]
        exp += [-E_with_maj_with_median(*J, n-1) + np.log(n),
                -E_with_maj_against_median(*J, n-1) + np.log(n),
                -E_not_with_maj_against_median(*J, n-1) + np.log(n),
                -E_with_maj_with_median(*J, n)]
        num, sign = fast_logsumexp(exp, coeffs)
        return sign * np.exp( num - logZ(*J) )

    def pk(*J, n=n):
        logpk = np.zeros(n-n//2)
        counter = 0
        for k in range(n//2+1, n-1):
            coeffs = [k * (k-1) / (n * (n-1)),
                      k * (n-k) / (n * (n-1)),
                      (n-k) * (n-k-1) / (n * (n-1)),
                      (n-k) * k / (n * (n-1))]
            exp = [-E_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                   -E_with_maj_against_median(*J, k) + np.log(binom(n,k)),
                   -E_not_with_maj_with_median(*J, k) + np.log(binom(n,k)),
                   -E_not_with_maj_against_median(*J, k) + np.log(binom(n,k))]
            logpk[counter] = fast_logsumexp(exp, coeffs)[0]
            counter += 1
        coeffs = [(n-2)/n, 1/n, 1/n]
        exp = [-E_with_maj_with_median(*J, n-1) + np.log(n),
               -E_with_maj_against_median(*J, n-1) + np.log(n),
               -E_not_with_maj_against_median(*J, n-1) + np.log(n)]
        logpk[counter] = fast_logsumexp(exp, coeffs)[0]
        counter += 1

        logpk[counter] = -E_with_maj_with_median(*J, n)
        return np.exp( logpk - logZ(*J) )

    return smo, smop, soo, sop, pk

def setup_mo_perturbation(n, Jmop, Jmo, Joop, Joo):
    """Perturbation of <sM sO>
    involves 4 couplings"""
    
    logPartitionList = []
    sMOpcoeffs = []
    sMOcoeffs = []
    sOOpcoeffs = []
    sOOcoeffs = []

    for k in range(n//2+1,n):
            # energy term determined by placement of median or Op voter in maj or min
            # consider median to be first spin
            # 0 indicates vote in maj
            # 1 indicates vote in min
            # coeffs are fraction of ways of arranging spins in maj and min coalitions given n choose k
            E = (-Jmop -Jmo*((k-2)-(n-k)) -Joop*((k-2)-(n-k))
                 -Joo*(binom(k-2,2)+binom(n-k,2)-(k-2)*(n-k)))
            coeff = k*(k-1) / (n*(n-1))
            sMOpcoeffs.append(1)
            sMOcoeffs.append(((k-2)-(n-k))/(n-2))
            sOOpcoeffs.append(((k-2)-(n-k))/(n-2))
            sOOcoeffs.append((binom(k-2,2)+binom(n-k,2)-(k-2)*(n-k))/binom(n-2,2))
            logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )

            E = (Jmop -Jmo*((k-1)-(n-k-1)) -Joop*((n-k-1)-(k-1))
                 -Joo*(binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1)))
            coeff = k*(n-k) / (n*(n-1))
            sMOpcoeffs.append(-1)
            sMOcoeffs.append(((k-1)-(n-k-1))/(n-2))
            sOOpcoeffs.append(((n-k-1)-(k-1))/(n-2))
            sOOcoeffs.append((binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1))/binom(n-2,2))
            logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )

            E = (Jmop -Jmo*((n-k-1)-(k-1)) -Joop*((k-1)-(n-k-1))
                 -Joo*(binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1)))
            coeff = k*(n-k) / (n*(n-1))
            if coeff>0:
                sMOpcoeffs.append(-1)
                sMOcoeffs.append(((n-k-1)-(k-1))/(n-2))
                sOOpcoeffs.append(((k-1)-(n-k-1))/(n-2))
                sOOcoeffs.append((binom(k-1,2)+binom(n-k-1,2)-(k-1)*(n-k-1))/binom(n-2,2))
                logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )

            E = (-Jmop -Jmo*((n-k-2)-k) -Joop*((n-k-2)-k)
                 -Joo*(binom(n-k-2,2)+binom(k,2)-(n-k-2)*k))
            coeff = (n-k)*(n-k-1) / (n*(n-1))
            if coeff>0:
                sMOpcoeffs.append(1)
                sMOcoeffs.append(((n-k-2)-k)/(n-2))
                sOOpcoeffs.append(((n-k-2)-k)/(n-2))
                sOOcoeffs.append((binom(n-k-2,2)+binom(k,2)-(n-k-2)*k)/binom(n-2,2))
                logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )

    # handle special case of k=n ====================================
    E = -Jmop -Jmo*(n-2) -Joop*(n-2) -Joo*binom(n-2,2)
    sMOpcoeffs.append(1)
    sMOcoeffs.append(1)
    sOOpcoeffs.append(1)
    sOOcoeffs.append(1)
    logPartitionList.append( -E )
    
    return (logPartitionList,
            sMOpcoeffs,
            sMOcoeffs,
            sOOpcoeffs,
            sOOcoeffs)

def setup_oo_perturbation(n, Jmop, Jmo, Jo1o2, Joop, Joo):
    """Perturbation of <sO_1 sO_2>
    involves 5 couplings"""
    
    logPartitionList = []
    sMOpcoeffs = []
    sMOcoeffs = []
    sO1O2pcoeffs = []
    sOOpcoeffs = []
    sOOcoeffs = []

    for k in range(n//2+1,n):
        # energy term determined by placement of median or Op voter in maj or min
        # consider median to be first spin
        # 0 indicates vote in maj
        # 1 indicates vote in min
        # coeffs are fraction of ways of arranging spins in maj and min coalitions given n choose k
        # 1
        E = (-2*Jmop -Jmo*(k-3-(n-k)) -Jo1o2 -Joo*(binom(k-3,2)+binom(n-k,2)-(k-3)*(n-k))
             -2*Joop*(k-3-(n-k)))
        coeff = k*(k-1)*(k-2) / (n*(n-1)*(n-2))
        sMOpcoeffs.append(1)
        sMOcoeffs.append((k-3-(n-k))/(n-3))
        sO1O2pcoeffs.append(1)
        sOOcoeffs.append((binom(k-3,2)+binom(n-k,2)-(k-3)*(n-k))/binom(n-3,2))
        sOOpcoeffs.append(2*(k-3-(n-k))/(2*n-6))
        logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )
        
        # 2
        E = (-Jmo*(k-2-(n-k-1)) +Jo1o2 -Joo*(binom(k-2,2)+binom(n-k-1,2)-(k-2)*(n-k-1))
             -Joop*((k-2-(n-k-1)) + (n-k-1-(k-2))))
        coeff = 2 * k*(k-1)*(n-k) / (n*(n-1)*(n-2))
        sMOpcoeffs.append(0)
        sMOcoeffs.append((k-2-(n-k-1))/(n-3))
        sO1O2pcoeffs.append(-1)
        sOOcoeffs.append((binom(k-2,2)+binom(n-k-1,2)-(k-2)*(n-k-1))/binom(n-3,2))
        sOOpcoeffs.append(((k-2-(n-k-1)) + (n-k-1-(k-2)))/(2*n-6))
        logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )

        # 3
        E = (2*Jmop -Jmo*(k-1-(n-k-2)) -Jo1o2 -Joo*(binom(k-1,2)+binom(n-k-2,2)-(k-1)*(n-k-2))
             -Joop*(n-k-2-(k-1))*2)
        coeff = k*(n-k)*(n-k-1) / (n*(n-1)*(n-2))
        if coeff>0:
            sMOpcoeffs.append(-1)
            sMOcoeffs.append((k-1-(n-k-2))/(n-3))
            sO1O2pcoeffs.append(1)
            sOOcoeffs.append((binom(k-1,2)+binom(n-k-2,2)-(k-1)*(n-k-2))/binom(n-3,2))
            sOOpcoeffs.append((n-k-2-(k-1))*2/(2*n-6))
            logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )
           
        # 4
        E = (2*Jmop -Jmo*(n-k-1-(k-2)) -Jo1o2 -Joo*(binom(k-2,2)+binom(n-k-1,2)-(k-2)*(n-k-1))
             -Joop*(k-2-(n-k-1))*2)
        coeff = k*(k-1)*(n-k) / (n*(n-1)*(n-2))
        sMOpcoeffs.append(-1)
        sMOcoeffs.append((n-k-1-(k-2))/(n-3))
        sO1O2pcoeffs.append(1)
        sOOcoeffs.append((binom(k-2,2)+binom(n-k-1,2)-(k-2)*(n-k-1))/binom(n-3,2))
        sOOpcoeffs.append((k-2-(n-k-1))*2/(2*n-6))
        logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )
        
        # 5
        E = (-Jmo*(n-k-2-(k-1)) +Jo1o2 -Joo*(binom(k-1,2)+binom(n-k-2,2)-(k-1)*(n-k-2))
             -Joop*((k-1-(n-k-2)) + (n-k-2-(k-1))))
        coeff = 2 * k*(n-k)*(n-k-1) / (n*(n-1)*(n-2))
        if coeff>0:
            sMOpcoeffs.append(0)
            sMOcoeffs.append((n-k-2-(k-1))/(n-3))
            sO1O2pcoeffs.append(-1)
            sOOcoeffs.append((binom(k-1,2)+binom(n-k-2,2)-(k-1)*(n-k-2))/binom(n-3,2))
            sOOpcoeffs.append(((k-1-(n-k-2)) + (n-k-2-(k-1)))/(2*n-6))
            logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )
        
        # 6
        E = (-2*Jmop -Jmo*(n-k-3-k) -Jo1o2 -Joo*(binom(n-k-3,2)+binom(k,2)-k*(n-k-3))
             -Joop*(n-k-3-k)*2)
        coeff = (n-k)*(n-k-1)*(n-k-2) / (n*(n-1)*(n-2))
        if coeff>0:
            sMOpcoeffs.append(1)
            sMOcoeffs.append((n-k-3-k)/(n-3))
            sO1O2pcoeffs.append(1)
            sOOcoeffs.append((binom(n-k-3,2)+binom(k,2)-k*(n-k-3))/binom(n-3,2))
            sOOpcoeffs.append((n-k-3-k)*2/(2*n-6))
            logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )
        
    # handle special case of k=n ====================================
    E = -2*Jmop -Jmo*(n-3) -Jo1o2 -Joo*binom(n-3,2) -2*Joop*(n-3)
    sMOpcoeffs.append(1)
    sMOcoeffs.append(1)
    sO1O2pcoeffs.append(1)
    sOOpcoeffs.append(1)
    sOOcoeffs.append(1)
    logPartitionList.append( -E )
    
    return logPartitionList, sMOpcoeffs, sMOcoeffs, sO1O2pcoeffs, sOOpcoeffs, sOOcoeffs

def setup_perturbation(J, n):
    """Full perturbation model as detailed on SCOTUS II pg. 124.
    
    Parameters
    ----------
    J : ndarray
        12 couplings specifying all perturbations possible for MVM.
    n : int
    
    Returns
    -------
    list
        Log of terms in partition function.
    list
        k voters in the majority for each term in partition list.
    list
        Coefficients of partition function necessary for calculating pairwise correlations.
    """
    
    logPartitionList = []
    kList = []
    sisjCoeffs = [[] for i in range(12)]
    
    for k in range(n//2+1,n):
        thiskCoeffSum = 0
        # energy term determined by placement of median or Op voter in maj or min
        # consider median to be first spin
        # 0 indicates vote in maj
        # 1 indicates vote in min
        # coeffs are fraction of ways of arranging spins in maj and min coalitions given n choose k
        # 1
        for config in bin_states(5):
            nmin = config.sum()  # no. of these five spins in the min.
            nmaj = 5 - nmin
#             if nmin<=(n-k):  # why is this condition unnecessary? because of nans
            # calculate coefficients for each coupling term
            couplingsCoeff = np.zeros(12)-1
            if config[0]==config[1]:
                couplingsCoeff[0] = 1.
            if config[0]==config[2]:
                couplingsCoeff[1] = 1.
            if config[1]==0:
                couplingsCoeff[2] = (k-nmaj) -(n-k-nmin)
            else:
                couplingsCoeff[2] = (n-k-nmin) -(k-nmaj)
            if config[2]==0:
                couplingsCoeff[3] = (k-nmaj) -(n-k-nmin)
            else:
                couplingsCoeff[3] = (n-k-nmin) -(k-nmaj)
            if config[1]==config[2]:
                couplingsCoeff[11] = 1

            if config[3]==config[4]:
                if config[1]==config[3]:
                    couplingsCoeff[4] = 2
                else:
                    couplingsCoeff[4] = -2
                if config[2]==config[3]:
                    couplingsCoeff[5] = 2
                else:
                    couplingsCoeff[5] = -2
                couplingsCoeff[6] = 1

                if config[3]==0:
                    couplingsCoeff[7] = 2 * (k-nmaj -(n-k-nmin))
                else:
                    couplingsCoeff[7] = -2 * (k-nmaj -(n-k-nmin))
                if config[0]==config[3]:
                    couplingsCoeff[10] = 2
                else:
                    couplingsCoeff[10] = -2
            else:
                couplingsCoeff[4] = 0
                couplingsCoeff[5] = 0
                couplingsCoeff[7] = 0
                couplingsCoeff[10] = 0
            couplingsCoeff[8] = binom(k-nmaj,2) + binom(n-k-nmin,2) -(k-nmaj)*(n-k-nmin)
            if config[0]==0:
                couplingsCoeff[9] = (k-nmaj) -(n-k-nmin)
            else:
                couplingsCoeff[9] = (n-k-nmin) -(k-nmaj)
            E = -J.dot(couplingsCoeff)
            
            # coefficient term in front of exponential counting multiplicity of state
            coeff = 1.
            majCounter = 0
            minCounter = 0
            for i in range(config.sum()):
                coeff *= n-k-minCounter
                minCounter += 1
            for i in range(config.sum(),5):
                coeff *= k-majCounter
                majCounter += 1
            coeff /= n*(n-1)*(n-2)*(n-3)*(n-4)

            if coeff>0:
                thiskCoeffSum += coeff
                kList.append(k)
                sisjCoeffs[0].append(couplingsCoeff[0])
                sisjCoeffs[1].append(couplingsCoeff[1])
                sisjCoeffs[2].append(couplingsCoeff[2]/(n-5))
                sisjCoeffs[3].append(couplingsCoeff[3]/(n-5))
                sisjCoeffs[4].append(couplingsCoeff[4]/2)
                sisjCoeffs[5].append(couplingsCoeff[5]/2)
                sisjCoeffs[6].append(couplingsCoeff[6])
                sisjCoeffs[7].append(couplingsCoeff[7]/2/(n-5))
                sisjCoeffs[8].append(couplingsCoeff[8]/binom(n-5,2))
                sisjCoeffs[9].append(couplingsCoeff[9]/(n-5))
                sisjCoeffs[10].append(couplingsCoeff[10]/2)
                sisjCoeffs[11].append(couplingsCoeff[11])
                # making this exact makes no difference
                #logPartitionList.append( -E +np.log(coeff) +np.log(comb(n,k,exact=True)) )
                logPartitionList.append( -E +np.log(coeff) +np.log(binom(n,k)) )
        # this should be 1
        # assert np.isclose( thiskCoeffSum, 1 )
        
    # handle special case of k=n ====================================
    couplingsCoeff = np.ones(12)
    couplingsCoeff[[2,3,9]] = n-5
    couplingsCoeff[[4,5,10]] = 2
    couplingsCoeff[7] = 2*(n-5)
    couplingsCoeff[8] = binom(n-5,2)
    E = -couplingsCoeff.dot(J)
    for i in sisjCoeffs:
        i.append(1)
    logPartitionList.append(-E)
    kList.append(n)
    sisjCoeffs = [np.array(coeffs) for coeffs in sisjCoeffs]
    
    return logPartitionList, kList, sisjCoeffs

def refine_perturbation(cost, J, refine_max_iter, refine_multiplier, tol):
    """This is useless.
    
    Simple iterative refinement of perturbation solution for couplings. This just
    increases the coupling if the pairwise correlation is too large and decreases it if
    the pairwise correlation is too small, multiplying the error by the refine_multiplier
    factor to change the coupling. When the error is already small, this can converge
    well.

    Parameters
    ----------
    cost : function
        Returns vector indicating the errors on the current parameter estimate.
    J : ndarray
    refine_max_iter : int
    refine_multiplier : float
    tol : float
        Norm2 error for stopping iteration.

    Returns
    -------
    ndarray
        Couplings.
    int
        Error flag. 0 is good, 1 means not converged.
    list
        History of errors (to make sure that errors are decaying and not growing or
        oscillating).
    """
    
    J = J.copy()
    dJ = np.ones_like(J)
    counter = 0
    errHistory = [1]
    while counter<refine_max_iter and errHistory[-1]>tol:
        dJ = -cost(J, True)
        errHistory.append(np.linalg.norm(dJ))
        J += dJ*refine_multiplier
        counter += 1
    if np.linalg.norm(dJ)<tol:
        errflag = 0
    else:
        errflag = 1
    errHistory.pop(0)
    return J, errflag, errHistory

def solve_mo_perturbation(n, J0,
                          eps=1e-4,
                          refine=True,
                          refine_multiplier=.2,
                          refine_max_iter=1000,
                          tol=1e-15,
                          full_output=False):
    """Solve for the gradient of the couplings w.r.t. a perturbation of type M->O.
    
    Parameters
    ----------
    n : int
    J0 : ndarray
        Initial guess.
    eps : float, 1e-4
    refine : bool, True
    refine_multiplier : float, .2
    refine_max_iter : int, 10_000
    tol : float, 1e-15
    full_output : bool, False
    
    Returns
    -------
    ndarray
        Estimate of derivative.
    dict from scipy.optimize.minimize (optional)
    list
        Error history for iterative algorithm.
    """
    
    smoExact = sum([(k/(n-1) - (n-1-k)/(n-1)) * binom(n-1,k)/2**(n-2)
                    for k in range((n-1)//2,n)])
    sooExact = 0.
    smopExact = smoExact*(1-eps) + eps
    smoExact = smoExact*(1-eps)
    
    def cost(params, as_vec=False):
        logPartitionList, smopC, smoC, soopC, sooC = setup_mo_perturbation(n, *params)
        smop = coeffs_to_corr(smopC, logPartitionList)
        smo = coeffs_to_corr(smoC, logPartitionList)
        soop = coeffs_to_corr(soopC, logPartitionList)
        soo = coeffs_to_corr(sooC, logPartitionList)
        if as_vec:
            return np.array([smop-smopExact,
                             smo-smoExact,
                             soop-sooExact,
                             soo-sooExact])
        return ((smop-smopExact)**2 + 
                (smo-smoExact)**2 + 
                (soop-sooExact)**2 +
                (soo-sooExact)**2)

    soln = minimize(cost, J0)
    if refine:
        soln = root(lambda x: cost(x, True), soln['x'])

    if full_output:
        return (soln['x']-J0)/eps, soln
    return (soln['x']-J0)/eps

def solve_oo_perturbation(n, J0,
                          eps=1e-4,
                          refine=True,
                          refine_multiplier=.2,
                          refine_max_iter=1000,
                          tol=1e-15,
                          full_output=False):
    """Solve for the gradient of the couplings w.r.t. a perturbation of type O->O.
    
    Parameters
    ----------
    n : int
    J0 : ndarray
        Initial guess.
    eps : float, 1e-4
    refine : bool, True
    refine_multiplier : float, .2
    refine_max_iter : int, 10_000
    tol : float, 1e-5
    full_output : bool, False
    
    Returns
    -------
    ndarray
        Estimate of derivative.
    dict from scipy.optimize.minimize (optional)
    list
        Error history for iterative algorithm.
    """
    
    smoExact = sum([(k/(n-1) - (n-1-k)/(n-1)) * binom(n-1,k)/2**(n-2)
                    for k in range((n-1)//2,n)])
    sooExact = 0.
    so1o2Exact = eps
    
    def cost(params, as_vec=False):
        logPartitionList, smopC, smoC, so1o2C, soopC, sooC = setup_oo_perturbation(n, *params)
        smop = coeffs_to_corr(smopC, logPartitionList)
        smo = coeffs_to_corr(smoC, logPartitionList)
        so1o2 = coeffs_to_corr(so1o2C, logPartitionList)
        soop = coeffs_to_corr(soopC, logPartitionList)
        soo = coeffs_to_corr(sooC, logPartitionList)
        if as_vec:
            return np.array([smop-smoExact,
                             smo-smoExact,
                             so1o2-so1o2Exact,
                             soop-sooExact,
                             soo-sooExact])
        return np.sqrt((smop-smoExact)**2 + 
                       (smo-smoExact)**2 + 
                       (so1o2-so1o2Exact)**2 +
                       (soop-sooExact)**2 +
                       (soo-sooExact)**2)

    soln = minimize(cost, J0)     
    if refine:
        soln = root(lambda x: cost(x, True), soln['x'])
    
    if full_output:
        return (soln['x']-J0)/eps, soln
    return (soln['x']-J0)/eps

def solve_om_perturbation(n, J0,
                          eps=1e-4,
                          refine=True,
                          refine_multiplier=.2,
                          refine_max_iter=1000,
                          tol=1e-15,
                          full_output=False):
    """Solve for the gradient of the couplings w.r.t. a perturbation of type O->M.
    
    Parameters
    ----------
    n : int
    J0 : ndarray
        Initial guess.
    eps : float, 1e-4
    refine : bool, True
    refine_multiplier : float, .2
    refine_max_iter : int, 10_000
    tol : float, 1e-5
    full_output : bool, False
    
    Returns
    -------
    ndarray
        Estimate of derivative.
    dict from scipy.optimize.minimize (optional)
    list
        Error history for iterative algorithm.
    """
    
    smoExact = sum([(k/(n-1) - (n-1-k)/(n-1)) * binom(n-1,k)/2**(n-2)
                    for k in range((n-1)//2,n)])
    sooExact = 0.
    smopExact = smoExact*(1-eps) + eps
    soopExact = smoExact*eps
    
    def cost(params, as_vec=False):
        logPartitionList, smopC, smoC, soopC, sooC = setup_mo_perturbation(n, *params)
        smop = coeffs_to_corr(smopC, logPartitionList)
        smo = coeffs_to_corr(smoC, logPartitionList)
        soop = coeffs_to_corr(soopC, logPartitionList)
        soo = coeffs_to_corr(sooC, logPartitionList)
        if as_vec:
            return np.array([smop-smopExact,
                             smo-smoExact,
                             soop-soopExact,
                             soo-sooExact])
        return np.sqrt((smop-smopExact)**2 + 
                       (smo-smoExact)**2 + 
                       (soop-soopExact)**2 +
                       (soo-sooExact)**2)

    soln = minimize(cost, J0)
    if refine:
        soln = root(lambda x: cost(x, True), soln['x'])
 
    if full_output:
        return (soln['x']-J0)/eps, soln
    return (soln['x']-J0)/eps

def setup_coupling_perturbations(n, Jpair, epsdJ=1e-3, refine_max_iter=10_000):
    """
    Parameters
    ----------
    n : int
    Jpair : ndarray
        Couplings between (M,O) and (O,O) for MVM.
    epsdJ : float, 1e-3

    Returns
    -------
    function
        _perturb_m_to_o
    function
        _perturb_o_to_m
    function
        _perturb_o_to_o
    """

    def _perturb_m_to_o(J, return_dJ=False, epsdJ=epsdJ, n=n):
        """Perturb couplings in direction given by M->O.

        Parameters
        ----------
        J : ndarray
            Couplings. Of size 12.
        return_dJ : bool, False

        Returns
        -------
        ndarray
            New couplings after applying perturbation. Of size 12 as indexed in SCOTUS II pg. 124. Can be 
            used directly in setup_perturbation().
        ndarray (optional)
            Return only the derivative instead of the new couplings moved along direction of gradient.
        """

        dJ = solve_mo_perturbation(n, [Jpair[0],Jpair[0],Jpair[1],Jpair[1]],
                                   refine_max_iter=refine_max_iter)
        J_ = J.copy()
        J_[0] += dJ[0]*epsdJ
        J_[[1,9,10]] += dJ[1]*epsdJ
        J_[[2,4,11]] += dJ[2]*epsdJ
        J_[[3,5,6,7,8]] += dJ[3]*epsdJ

        if return_dJ:
            dJ_ = np.zeros_like(J)
            dJ_[0] = dJ[0]
            dJ_[[1,9,10]] = dJ[1]
            dJ_[[2,4,11]] = dJ[2]
            dJ_[[3,5,6,7,8]] = dJ[3]
            return J_, dJ_
        return J_

    def _perturb_o_to_m(J, return_dJ=False, epsdJ=epsdJ, n=n):
        """Perturb couplings in direction given by O->M.

        Parameters
        ----------
        J : ndarray
            Couplings. Of size 12.
        return_dJ : bool, False

        Returns
        -------
        ndarray
            New couplings after applying perturbation. Of size 12 as indexed in SCOTUS II pg. 124. Can be 
            used directly in setup_perturbation().
        ndarray (optional)
            Return only the derivative instead of the new couplings moved along direction of gradient.
        """

        dJ = solve_om_perturbation(n, [Jpair[0],Jpair[0],Jpair[1],Jpair[1]],
                                   refine_max_iter=refine_max_iter)
        J_ = J.copy()
        J_[0] += dJ[0]*epsdJ
        J_[[1,9,10]] += dJ[1]*epsdJ
        J_[[2,4,11]] += dJ[2]*epsdJ
        J_[[3,5,6,7,8]] += dJ[3]*epsdJ

        if return_dJ:
            dJ_ = np.zeros_like(J)
            dJ_[0] = dJ[0]
            dJ_[[1,9,10]] = dJ[1]
            dJ_[[2,4,11]] = dJ[2]
            dJ_[[3,5,6,7,8]] = dJ[3]
            return J_, dJ_
        return J_

    def _perturb_o_to_o(J, return_dJ=False, epsdJ=epsdJ, n=n):
        """Perturb couplings in direction given by O->O.

        Parameters
        ----------
        J : ndarray
            Couplings. Of size 12.
        return_dJ : bool, False

        Returns
        -------
        ndarray
            New couplings after applying perturbation. Of size 12 as indexed in SCOTUS II pg. 124. Can be 
            used directly in setup_perturbation().
        ndarray (optional)
            Return only the derivative instead of the new couplings moved along direction of gradient.
        """

        dJ = solve_oo_perturbation(n, [Jpair[0],Jpair[0],Jpair[1],Jpair[1],Jpair[1]],
                                   refine_max_iter=refine_max_iter)
        J_ = J.copy()
        J_[10] += dJ[0]*epsdJ
        J_[[0,1,9]] += dJ[1]*epsdJ
        J_[6] += dJ[2]*epsdJ
        J_[[4,5,7]] += dJ[3]*epsdJ
        J_[[2,3,8,11]] += dJ[4]*epsdJ

        if return_dJ:
            dJ_ = np.zeros_like(J)
            dJ_[10] = dJ[0]
            dJ_[[0,1,9]] = dJ[1]
            dJ_[6] = dJ[2]
            dJ_[[4,5,7]] = dJ[3]
            dJ_[[2,3,8,11]] = dJ[4]
            return J_, dJ_
        return J_
    
    return _perturb_m_to_o, _perturb_o_to_m, _perturb_o_to_o

def square_J(J, n):
    """Convert vector form of couplings to given to setup_perturbation() into square matrix 
    for use with ConIII module.
    
    Parameters
    ----------
    J : ndarray
        12 couplings ordered as passed into setup_perturbation().
    n : int
        System size.
        
    Returns
    -------
    ndarray
        Square couplings matrix.
    """
    
    Jmat = np.zeros((n,n))
    Jmat[0,1] = J[0]
    Jmat[0,2] = J[1]
    Jmat[1,5:] = J[2]
    Jmat[2,5:] = J[3]
    Jmat[1,3] = Jmat[1,4] = J[4]
    Jmat[2,3] = Jmat[2,4] = J[5]
    Jmat[3,4] = J[6]
    Jmat[3,5:] = Jmat[4,5:] = J[7]
    for j in range(5,n):
        Jmat[j,j+1:] = J[8]
    Jmat[0,5:] = J[9]
    Jmat[0,3] = Jmat[0,4] = J[10]
    Jmat[1,2] = J[11]
    Jmat += Jmat.T
    return Jmat

def coeffs_to_corr(coeffs, logPartitionList):
    num = fast_logsumexp(logPartitionList, coeffs)
    return num[1] * np.exp(num[0] - fast_logsumexp(logPartitionList)[0])

def coeffs_to_logcorr(coeffs, logPartitionList):
    num = fast_logsumexp(logPartitionList, coeffs)
    return num[0] - fast_logsumexp(logPartitionList)[0]

def expand_small_fim(smallfim, n):
    """Populate the full FIM of dimensions (N*(N-1), N*(N-1)).
    
    Parameters
    ----------
    smallfim : ndarray
        Should be of dimensions 3x3 or tuple of diag entries then off diag.
    n : int
        System size.
    
    Returns
    -------
    ndarray
    """
    
    assert smallfim.shape==(3,3) or type(smallfim) is tuple
    if type(smallfim) is tuple:
        smallfim_ = squareform(smallfim[1])
        smallfim_[np.diag_indices(3)] = smallfim[0]
        smallfim = smallfim_

    fim = np.zeros((n*(n-1),n*(n-1)))
    # (m->0)^2
    fim[:n-1,:n-1] = smallfim[0,0]
    for i in range(1,n):
        # m->o, o->m
        fim[:n-1,i*(n-1)] = fim[i*(n-1),:n-1] = smallfim[0,1]
        fim[:n-1,i*(n-1)+1:(i+1)*(n-1)] = fim[i*(n-1)+1:(i+1)*(n-1),:n-1] = smallfim[0,2]
    for i in range(1,n):
        for j in range(1,n):
            # (o->m)^2
            fim[i*(n-1),j*(n-1)] = smallfim[1,1]
            # o->m, o->o
            fim[i*(n-1),j*(n-1)+1:(j+1)*(n-1)] = fim[j*(n-1)+1:(j+1)*(n-1),i*(n-1)] = smallfim[1,2]
            # (o->o)^2
            fim[i*(n-1)+1:(i+1)*(n-1),j*(n-1)+1:(j+1)*(n-1)] = smallfim[2,2]
    return fim

def logZ_to_Ek_pk(logPartitionList, kList):
    """Use list of terms in partition function to calculate the energies and probabilities
    of states.
    """
    n = max(kList)
    Ek = np.array([-fast_logsumexp([logPartitionList[i] for i in range(len(kList)) if kList[i]==k])[0]
                  for k in range(n//2+1,n+1)])
    logZ = fast_logsumexp(logPartitionList)[0]
    pk = np.exp([fast_logsumexp([logPartitionList[i] for i in range(len(kList)) if kList[i]==k])[0] - logZ
                for k in range(n//2+1,n+1)])
    return Ek, pk

def diag_fim(n, J_, Ek, pk, epsdJ):
    """Diagonal elements of the FIM where only a single correlation term is perturbed."""
    logPartitionList, kList, sisjCoeffs = setup_perturbation(J_, n)
    Eknew, pknew = logZ_to_Ek_pk(logPartitionList, kList)
    dE = Eknew-Ek
    return (pk.dot(dE**2)-dE.dot(pk)**2) / epsdJ**2 / np.log(2)

def off_diag_fim(n, newJ1, newJ2, Ek, pk, epsdJ):
    """Off-diagonal elements of the FIM where pairs of correlation term are perturbed."""
    logPartitionList, kList, sisjCoeffs = setup_perturbation(newJ1, n)
    Eknew, pknew = logZ_to_Ek_pk(logPartitionList, kList)
    dE1 = Eknew-Ek
    # check for numerical precision errors
    assert np.isclose(pknew.sum(),1)
    
    logPartitionList, kList, sisjCoeffs = setup_perturbation(newJ2, n)
    Eknew, pknew = logZ_to_Ek_pk(logPartitionList, kList)
    dE2 = Eknew-Ek
    # check for numerical precision errors
    assert np.isclose(pknew.sum(),1)
    
    return pk.dot((dE1-dE1.dot(pk))*(dE2-dE2.dot(pk))) / epsdJ**2 / np.log(2)

def _fim(n):
    """FIM for the MVM. Hard way of calculating it using IsingFisherCurvatureMethod2.

    This only allows us to check if we are computing the Hessian properly once we already
    have the change in couplings. If you wish to check whether the change in the couplings
    are correct you should check test_mvm.py where the correlations are perturbed and the
    inverse Iisng problem solved.

    Parameters
    ----------
    n : int

    Returns
    -------
    ndarray
        fim
    ndarray
        eigval
    ndarray
        eigvec
    """
    
    from .fim import IsingFisherCurvatureMethod2

    Jpair = couplings(n)
    # map the couplings to the full perturbation scheme
    J = np.zeros(12)
    J[[0,1,9,10]] = Jpair[0]
    J[J==0] = Jpair[1]

    _perturb_m_to_o, _perturb_o_to_m, _perturb_o_to_o = setup_coupling_perturbations(n, Jpair)

    # Use IsingFisherCurvatureMethod2 to calculate FIM quickly.
    isingdkl = IsingFisherCurvatureMethod2(n,
                                           h=np.zeros(n),
                                           J=squareform(square_J(J,n)),
                                           precompute=False)
    dJ1 = np.insert(squareform(square_J(_perturb_m_to_o(J, True)[1],n)), 0, np.zeros(n))
    dJ2 = np.insert(squareform(square_J(_perturb_o_to_m(J, True)[1],n)), 0, np.zeros(n))
    dJ3 = np.insert(squareform(square_J(_perturb_o_to_o(J, True)[1],n)), 0, np.zeros(n))

    isingdkl.dJ = np.vstack((dJ1,dJ2,dJ3))
    smallfim = isingdkl.maj_curvature(epsdJ=1e-5, iprint=False)
    fim = expand_small_fim(smallfim, n)

    eigval, eigvec = np.linalg.eig(fim)
    sortix = np.argsort(eigval)[::-1][:2]
    eigval = eigval[sortix].real
    eigvec = eigvec[:,sortix].real
    return fim, eigval, eigvec

def fim(n, epsdJ=1e-5):
    """FIM for the MVM.

    Parameters
    ----------
    n : int
    epsdJ : float, 1e-5

    Returns
    -------
    ndarray
        FIM.
    ndarray
        The two eigenvalues sorted.
    ndarray
        Eigvectors sorted.
    """
    
    # setup solution
    # Couplings for MVM.
    Jpair = couplings(n)
    _perturb_m_to_o, _perturb_o_to_m, _perturb_o_to_o = setup_coupling_perturbations(n, Jpair, epsdJ)

    # map the couplings to the full perturbation scheme
    J = np.zeros(12)
    J[[0,1,9,10]] = Jpair[0]
    J[J==0] = Jpair[1]
    Jmo = _perturb_m_to_o(J)
    Jom = _perturb_o_to_m(J)
    Joo = _perturb_o_to_o(J)

    # p(k) for the MVM without perturbation
    logPartitionList, kList, sisjCoeffs = setup_perturbation(J, n)
    Ek, pk = logZ_to_Ek_pk(logPartitionList, kList)
    assert np.isclose(pk.sum(), 1)
    
    # calculation entries of the FIM
    # fill diagonal elements
    smallfim = np.zeros((3,3))
    smallfim[0,0] = diag_fim(n, Jmo, Ek, pk, epsdJ)
    smallfim[1,1] = diag_fim(n, Jom, Ek, pk, epsdJ)
    smallfim[2,2] = diag_fim(n, Joo, Ek, pk, epsdJ)

    smallfim[0,1] = smallfim[1,0] = off_diag_fim(n, Jmo, Jom, Ek, pk, epsdJ)
    smallfim[0,2] = smallfim[2,0] = off_diag_fim(n, Jmo, Joo, Ek, pk, epsdJ)
    smallfim[1,2] = smallfim[2,1] = off_diag_fim(n, Joo, Jom, Ek, pk, epsdJ)

    fim = expand_small_fim(smallfim, n)

    eigval, eigvec = np.linalg.eigh(fim)
    sortix = np.argsort(eigval)[::-1][:2]
    eigval = eigval[sortix].real
    eigvec = eigvec[:,sortix].real
    return smallfim, eigval, eigvec, pk
