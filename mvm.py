# =============================================================================================== #
# Module for Median Voter Model.
# Author: Eddie Lee, edl56@cornell.edu
# =============================================================================================== #
import numpy as np
from coniii.utils import *
from coniii.enumerate import fast_logsumexp


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

def couplings(n, data_corr=None, full_output=False, tol=1e-12, max_refine_iter=1000):
    """Find couplings corresponding to mvm pairwise correlations numerically.

    Parameters
    ----------
    n : int
    data_corr : ndarray, None
    full_output : bool, False

    Returns
    -------
    ndarray
        [Jmo, Joo]
    """
    
    from scipy.optimize import minimize
    
    if data_corr is None:
        smo, soo = corr(n)
        smo_fun, _, soo_fun, _, _ = setup_maxent(n)
        def cost(params):
            Jmo, Joo = params
            return np.sqrt((smo-smo_fun(Jmo, Jmo, Joo, Joo))**2 +
                           (soo-soo_fun(Jmo, Jmo, Joo, Joo))**2)
        soln = minimize(cost, [0,0], tol=1e-10)
    else:
        smo, smop, soo, soop = data_corr
        smo_fun, smop_fun, soo_fun, soop_fun, _ = setup_maxent(n)
        def cost(params):
            return np.sqrt((smo - smo_fun(*params))**2 +
                           (smop - smop_fun(*params))**2 + 
                           (soo - soo_fun(*params))**2 +
                           (soop - soop_fun(*params))**2)
        soln = minimize(cost, [.1,.1,0,0], tol=1e-10)

        # refine solutions
        def each_err(params):
            return -np.array([smo - smo_fun(*params),
                              smop - smop_fun(*params),
                              soo - soo_fun(*params),
                              soop - soop_fun(*params)])
        # simple algorithm with inertia (using average of last nAvg points for error estimation)
        counter = 0
        err = prevErr = cost(soln['x'])
        prevJ = soln['x'].copy()
        eps = 1
        nAvg = 10
        cumerr = err*nAvg
        while counter<max_refine_iter and err>tol and eps>tol:
            if (counter%nAvg)==0:
                if np.sqrt(cumerr/nAvg)>prevErr:
                    # undo
                    soln['x'] = prevJ.copy()
                    eps /= 2
                else:
                    prevErr = np.sqrt(cumerr/nAvg)
                    prevJ = soln['x'].copy()
                    eps *= 1.5
                cumerr = 0

            dJ = -each_err(soln['x']) * eps
            soln['x'] += dJ
            err = cost(soln['x'])
            cumerr += err**2
            counter += 1
        # simplest iterative algorithm
        #counter = 0
        #err = prevErr = cost(soln['x'])
        #prevJ = soln['x'].copy()
        #eps = 1
        #seqSuccess = 0
        #while counter<max_refine_iter and err>tol and eps>tol:
        #    dJ = -each_err(soln['x']) * eps
        #    soln['x'] += dJ

        #    err = cost(soln['x'])
        #    if err>prevErr:
        #        soln['x'] -= dJ
        #        eps /= 2
        #        seqSuccess = 0
        #    else:
        #        prevErr = err
        #        seqSuccess += 1
        #        if seqSuccess==10:  # every 10 successful steps, update eps
        #            eps *= 1.5
        #            seqSuccess = 0
        #    counter += 1
        print(counter,eps,err)
    if full_output:
        return soln['x'], soln
    return soln['x']

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
