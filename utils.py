# ============================================================================================ #
# Quick access to useful modules from SCOTUS project.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ============================================================================================ # 
import numpy as np
from numba import njit
from coniii.utils import *


def solve_linearized_pair_perturbation(n, iStar, aStar, sisj, p,
                                       allStates=None,
                                       eps=1e-6,
                                       check_stability=True,
                                       full_output=False):
    """Only considering pairs as <si*sj> is converted into <si*sk>.
    
    Find linear perturbation for couplings when spin iStar is fractionally converted
    into spin aStar. This is equivalent to a linear transformation of every one of iStar's
    correlations into aStar's correlations at small rate eps. Or sampling randomly iStar's
    orientation and replacing with aStar's with probability eps.
    
    Solve following system of equations.
    A.dot(delta_{kl}) = C
    
    Parameters
    ----------
    n : int
    iStar : int
        Index of spin to be replaced.
    aStar : int
        Index of spin to replace with.
    sisj : ndarray
        Pairwise correlations.
    p : ndarray
    allStates : ndarray, None
        {-1,1} basis.
    eps : float, 1e-6
        Step size for approximating derivative.
    check_stability : bool, True
        Check whether or not linearization is accurate by halving eps.
    full_output : False
    
    Returns
    -------
    ndarray
        Jacobian of couplings wrt eps parameter titrating votes.
    """
    
    assert sisj.size==(n*(n-1)//2)
    if allStates is None:
        allStates = bin_states(n, True)

    A = np.zeros((n*(n-1)//2,n*(n-1)//2))
    C = np.zeros(n*(n-1)//2)
    for ijCount,(i,j) in enumerate(combinations(range(n),2)):
        if i==iStar:
            if aStar<j:
                C[ijCount] = eps*(sisj[unravel_index((aStar,j),n)] - sisj[unravel_index((i,j),n)])
            elif aStar>j:
                C[ijCount] = eps*(sisj[unravel_index((j,aStar),n)] - sisj[unravel_index((i,j),n)])
            else:
                C[ijCount] = eps*(1-sisj[unravel_index((i,j),n)])
        elif j==iStar:
            if aStar<i:
                C[ijCount] = eps*(sisj[unravel_index((aStar,i),n)] - sisj[unravel_index((i,j),n)])
            elif aStar>i:
                C[ijCount] = eps*(sisj[unravel_index((i,aStar),n)] - sisj[unravel_index((i,j),n)])
            else:
                C[ijCount] = eps*(1-sisj[unravel_index((i,j),n)])
        # this is just a copy of sisj if nothing changes
        newsisj = C[ijCount] + sisj[unravel_index((i,j),n)]

        for klCount,(k,l) in enumerate(combinations(range(n),2)):
            A[ijCount,klCount] = np.prod(allStates[:,(i,j,k,l)],1).dot(p) - newsisj*sisj[klCount]
    
    # factor out linear dependence on eps
    dJ = np.linalg.solve(A,C)/eps

    if check_stability:
        # double epsilon and make sure solution does not change by a large amount
        dJtwiceEps = solve_linearized_pair_perturbation(n, iStar, aStar, sisj, p,
                                                        eps=eps/2,
                                                        allStates=allStates,
                                                        check_stability=False)
        # print if relative change is more than .1% for any entry
        if ((np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ)))>-3).any():
            print("Unstable solution. Recommend shrinking eps.")
    if full_output:
        return dJ, (A, C)
    return dJ

def tweak_constraints(constraints, i, j, theta, n):
    """Replace all of i's correlations with j's correlations with weighting theta.
    
    Parameters
    ----------
    constraints : ndarray
        Concatenated list of h and J.
    i : int
        Spin to replace.
    j : int
        Spin to replace with.
    theta : ndarray
        Probability of replacement.
    n : int
        System size.
        
    Returns
    -------
    ndarray
        Updated constraints.
    """
    
    from coniii.utils import unravel_index
    newconstraints = constraints.copy()
    
    # means
    newconstraints[i] = constraints[j]*theta + constraints[i]*(1-theta)
    
    # pairwise correlations
    for k in np.delete(range(n), [i,j]):
        if i<k:
            ik=unravel_index((i,k),n)
        else:
            ik=unravel_index((k,i),n)
        if j<k:
            jk=unravel_index((j,k),n)
        else:
            jk=unravel_index((k,j),n)
        if i<j:
            ij=unravel_index((i,j),n)
        else:
            ij=unravel_index((j,i),n)
        
        newconstraints[ik+n]=constraints[ik+n]*(1-theta) + theta*constraints[jk+n]
        newconstraints[ij+n]=constraints[ij+n]*(1-theta) + theta
    return newconstraints

def define_energy_basin_functions(calc_observables):
    """Define functions for finding energy basins. Need calc_observables() to be defined in global namespace.

    Parameters
    ----------
    calc_observables : function

    Returns
    -------
    function
        find_energy_basin
    function
        flip_least_stable_spin
    """

    def find_energy_basin(x, hJ):
        """
        Find the local energy minimum.
        """

        x = x.copy()
        xprev = np.zeros(len(x), dtype=int)
        while (x!=xprev).any():
            xprev = x.copy()
            x = flip_least_stable_spin(x, hJ)
        return x
    
    def flip_least_stable_spin(x, hJ, calc_observables=calc_observables):
        """
        Flip the least stable spin.
        """

        E = -calc_observables(x[None,:]).dot(hJ)
        dE = np.zeros(x.size)
        for i in range(x.size):
            x[i] *= -1
            dE[i] = -calc_observables(x[None,:]).dot(hJ)-E
            x[i] *= -1

        if (dE<0).any():
            x[np.argmin(dE)] *= -1
        return x 

    return find_energy_basin, flip_least_stable_spin
