# ============================================================================================ #
# Quick access to useful modules from SCOTUS project.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ============================================================================================ # 
import numpy as np
from numba import njit
from coniii.utils import *
import importlib
from warnings import warn
from itertools import combinations
np.seterr(divide='ignore')


# ==================
# Functions
# ==================
@njit
def factorial(x):
    f = 1.
    while x>0:
        f *= x
        x -= 1
    return f

@njit
def binom(n,k):
    return factorial(n)/factorial(n-k)/factorial(k)

@njit
def jit_all(x):
    for x_ in x:
        if not x_:
            return False
    return True

@njit
def unravel_index(ijk, n):
    """Unravel multi-dimensional index to flattened index but specifically for
    multi-dimensional analog of an upper triangular array (lower triangle indices are not
    counted).

    Parameters
    ----------
    ijk : tuple
        Raveled index to unravel.
    n : int
        System size.

    Returns
    -------
    ix : int
        Unraveled index.
    """
    
    if len(ijk)==1:
        raise Exception

    assert jit_all([ijk[i]<ijk[i+1] for i in range(len(ijk)-1)])
    assert jit_all([i<n for i in ijk])

    ix = np.sum(np.array([int(binom(n-1-i,len(ijk)-1)) for i in range(ijk[0])]))
    for d in range(1, len(ijk)-1):
        if (ijk[d]-ijk[d-1])>1:
            ix += np.sum(np.array([int(binom(n-i-1,len(ijk)-d-1)) for i in range(ijk[d-1]+1, ijk[d])]))
    ix += ijk[-1] -ijk[-2] -1
    return ix


# ======= #
# Classes #
# ======= #
class IsingFisherCurvatureMethod1():
    def __init__(self, n, h=None, J=None, eps=1e-7, precompute=True, n_cpus=None):
        """
        Parameters
        ----------
        n : int
        h : ndarray, None
        J : ndarray, None
        eps : float, 1e-7
        precompute : bool, True
        n_cpus : int, None
        """
        
        import multiprocess as mp

        assert n>1 and 0<eps<.1
        self.n = n
        self.eps = eps
        self.hJ = np.concatenate((h,J))

        self.ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        self.sisj = self.ising.calc_observables(self.hJ)
        self.p = self.ising.p(self.hJ)
        self.allStates = bin_states(n, True).astype(np.int8)
        
        n_cpus = n_cpus or mp.cpu_count()
        self.pool = mp.Pool(n_cpus)

        # cache triplet and quartet products
        self.triplets = {}
        self.quartets = {}
        for i in range(n):
            for j,k in combinations(range(n),2):
                self.triplets[(i,j,k)] = np.prod(self.allStates[:,(i,j,k)],1).astype(np.int8)
        for i,j in combinations(range(n),2):
            for k in range(n):
                self.triplets[(i,j,k)] = np.prod(self.allStates[:,(i,j,k)],1).astype(np.int8)
            for k,l in combinations(range(n),2):
                self.quartets[(i,j,k,l)] = np.prod(self.allStates[:,(i,j,k,l)],1).astype(np.int8)
    
        if precompute:
            self.dJ = self.compute_dJ()
        else:
            self.dJ = np.zeros((self.n,self.n+(self.n-1)*self.n//2))

    def compute_dJ(self, p=None, sisj=None):
        # precompute linear change to parameters for small perturbation
        dJ = np.zeros((self.n,self.n+(self.n-1)*self.n//2))
        for i in range(self.n):
            dJ[i], errflag = self.solve_linearized_perturbation(i, p=p, sisj=sisj)
        return dJ

    def observables_after_perturbation(self, i,
                                       eps=None,
                                       perturb_up=True):
        """Perturb all specified spin by forcing it point upwards with probability eps/2.
        Perturb the corresponding mean and the correlations with other spins j.
        
        Parameters
        ----------
        i : int
        eps : float, None
        perturb_up : bool, True
            If True, make the specified spin point up +1. If False, make it point down -1.

        Returns
        -------
        ndarray
            Observables <si> and <sisj> after perturbation.
        """
        
        if not hasattr(i,'__len__'):
            i = (i,)
        if not hasattr(eps,'__len__'):
            eps = eps or self.eps
            eps = [eps]*len(i)
        n = self.n
        si = self.sisj[:n]
        sisj = self.sisj[n:]
       
        siNew = si.copy()
        sisjNew = sisj.copy()
        
        if perturb_up:
            for i_,eps_ in zip(i,eps):
                # observables after perturbations
                self._observables_after_perturbation_up(siNew, sisjNew, i_, eps_)
        else:
            for i_,eps_ in zip(i,eps):
                # observables after perturbations
                self._observables_after_perturbation_down(siNew, sisjNew, i_, eps_)

        return np.concatenate((siNew, sisjNew))
   
    def _observables_after_perturbation_up(self, si, sisj, i, eps):
        """        
        Parameters
        ----------
        si : ndarray
        sisj : ndarray
        i : int
        eps : float
        """

        n = self.n
        
        # observables after perturbations
        si[i]  = (1-eps)*si[i] + eps

        for j in np.delete(range(n),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)
            sisj[ijix] = (1-eps)*sisj[ijix] + eps*si[j]

    def _observables_after_perturbation_down(self, si, sisj, i, eps):
        """        
        Parameters
        ----------
        si : ndarray
        sisj : ndarray
        i : int
        eps : float
        """

        n = self.n
        
        # observables after perturbations
        si[i]  = (1-eps)*si[i] - eps

        for j in np.delete(range(n),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)
            sisj[ijix] = (1-eps)*sisj[ijix] - eps*si[j]

    def _solve_linearized_perturbation(self, iStar, eps=None, perturb_up=True):
        """Consider a perturbation to a single spin.
        
        Parameters
        ----------
        iStar : int
        eps : float, None
        perturb_up : bool, True

        Returns
        -------
        ndarray
            Linear change in maxent parameters for given iStar.
        """
        
        from coniii.solvers import Enumerate

        n = self.n
        p = self.p
        if eps is None:
            eps = self.eps
        C = self.observables_after_perturbation(iStar, eps=eps, perturb_up=perturb_up)

        solver = Enumerate(n, calc_observables_multipliers=self.ising.calc_observables)
        if perturb_up:
            return (solver.solve(C)-self.hJ)/eps

        # account for sign of perturbation on fields
        dJ = (solver.solve(C)-self.hJ)/eps
        if not perturb_up:
            dJ *= -1
        return dJ

    def solve_linearized_perturbation(self, iStar,
                                      p=None,
                                      sisj=None,
                                      full_output=False,
                                      eps=None,
                                      check_stability=True,
                                      method='inverse'):
        """Consider a perturbation to a single spin.
        
        Parameters
        ----------
        iStar : int
        p : ndarray, None
        sisj : ndarray, None
        full_output : bool, False
        eps : float, None
        check_stability : bool, False
        method : str, 'inverse'
            Can be 'inverse' or 'lstsq'

        Returns
        -------
        ndarray
            dJ
        int
            Error flag. Returns 0 by default. 1 means badly conditioned matrix A.
        tuple (optional)
            (A,C)
        """
        
        eps = eps or self.eps
        n = self.n
        if p is None:
            p = self.p
        if sisj is None:
            si = self.sisj[:n]
            sisj = self.sisj[n:]
        else:
            si = sisj[:n]
            sisj = sisj[n:]
        A = np.zeros((n+n*(n-1)//2, n+n*(n-1)//2))
        C = self.observables_after_perturbation(iStar, eps=eps, perturb_up=si[iStar]<0)
        
        # mean constraints
        for i in range(n):
            for k in range(n):
                if i==k:
                    A[i,i] = 1 - C[i]*si[i]
                else:
                    if i<k:
                        ikix = unravel_index((i,k),n)
                    else:
                        ikix = unravel_index((k,i),n)
                    A[i,k] = sisj[ikix] - C[i]*si[k]

            for klcount,(k,l) in enumerate(combinations(range(n),2)):
                A[i,n+klcount] = self.triplets[(i,k,l)].dot(p) - C[i]*sisj[klcount]
        
        # pair constraints
        for ijcount,(i,j) in enumerate(combinations(range(n),2)):
            for k in range(n):
                A[n+ijcount,k] = self.triplets[(i,j,k)].dot(p) - C[n+ijcount]*si[k]
            for klcount,(k,l) in enumerate(combinations(range(n),2)):
                A[n+ijcount,n+klcount] = self.quartets[(i,j,k,l)].dot(p) - C[n+ijcount]*sisj[klcount]
    
        C -= self.sisj
        if method=='inverse':
            # factor out linear dependence on eps
            try:
                dJ = np.linalg.solve(A,C)/eps
            except np.linalg.LinAlgError:
                dJ = np.zeros(C.size)+np.nan
        else:
            dJ = np.linalg.lstsq(A,C)[0]/eps
        if si[iStar]>=0:
            dJ *= -1

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps, errflag = self.solve_linearized_perturbation(iStar,
                                                                     eps=eps/2,
                                                                     check_stability=False,
                                                                     p=p,
                                                                     sisj=np.concatenate((si,sisj)))
            # print if relative change is more than .1% for any entry
            if ((np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ)))>-3).any():
                print("Unstable solution. Recommend shrinking eps.")
                   
        if np.linalg.cond(A)>1e15:
            warn("A is badly conditioned.")
            errflag = 1
        else:
            errflag = 0
        if full_output:
            return dJ, errflag, (A, C)
        return dJ, errflag
    
    def dkl_curvature(self, *args, **kwargs):
        """Wrapper for _dkl_curvature() to find best finite diff step size."""

        if not 'epsdJ' in kwargs.keys():
            kwargs['epsdJ'] = 1e-4
        if not 'check_stability' in kwargs.keys():
            kwargs['check_stability'] = True
        if 'full_output' in kwargs.keys():
            full_output = kwargs['full_output']
        else:
            full_output = False
        kwargs['full_output'] = True
        epsDecreaseFactor = 10
        
        converged = False
        prevHess, errflag, prevNormerr = self._dkl_curvature(*args, **kwargs)
        kwargs['epsdJ'] /= epsDecreaseFactor
        while (not converged) and errflag:
            hess, errflag, normerr = self._dkl_curvature(*args, **kwargs)
            # end loop if error starts increasing again
            if errflag and normerr<prevNormerr:
                prevHess = hess
                prevNormerr = normerr
                kwargs['epsdJ'] /= epsDecreaseFactor
            else:
                converged = True
        if not converged and not errflag:
            normerr = None
        hess = prevHess
        
        if full_output:
            return hess, errflag, normerr
        return hess

    def _dkl_curvature(self,
                      hJ=None,
                      dJ=None,
                      epsdJ=1e-4,
                      n_cpus=None,
                      check_stability=False,
                      rtol=1e-3,
                      zero_out_small_p=True,
                      p_threshold=1e-15,
                      full_output=False):
        """Calculate the hessian of the KL divergence (Fisher information metric) w.r.t.
        the theta_{ij} parameters replacing the spin i by sampling from j.

        Use single step finite difference method to estimate Hessian.
        
        Parameters
        ----------
        hJ : ndarray, None
            Ising model parameters.
        dJ : ndarray, None
            Linear perturbations in parameter space corresponding to Hessian at given hJ.
            These can be calculuated using self.solve_linearized_perturbation().
        epsdJ : float, 1e-4
            Step size for taking linear perturbation wrt parameters.
        n_cpus : int, None
        check_stability : bool, False
        rtol : float, 1e-3
            Relative tolerance for each entry in Hessian when checking stability.
        zero_out_small_p : bool, True
            If True, set all small values below p_threshold to 0.
        p_threshold : float, 1e-15
        full_output : bool, False
            
        Returns
        -------
        ndarray
            Hessian.
        int (optional)
            Error flag. 1 indicates rtol was exceeded. None indicates that no check was
            done.
        float (optional)
            Norm difference between hessian with step size eps and eps/2.
        """
        
        n = self.n
        if hJ is None:
            hJ = self.hJ
            p = self.p
        else:
            p = self.ising.p(hJ)
        log2p = np.log2(p)
        if dJ is None:
            dJ = self.dJ

        if zero_out_small_p:
            log2p[p<p_threshold] = -np.inf
            p = p.copy()
            p[p<p_threshold] = 0.
        
        # diagonal entries
        def diag(i, hJ=hJ, ising=self.ising, dJ=dJ, p=p):
            newhJ = hJ.copy()
            newhJ += dJ[i]*epsdJ
            modp = ising.p(newhJ)
            return np.nansum(2*(log2p-np.log2(modp))*p) / epsdJ**2
            
        # Compute off-diagonal entries. These don't account for the subtraction of the
        # diagonal elements which are removed later To see this, expand D(theta_i+del,
        # theta_j+del) to second order.
        def off_diag(args, hJ=hJ, ising=self.ising, dJ=dJ, p=p):
            i, j = args
            newhJ = hJ.copy()
            newhJ += (dJ[i]+dJ[j])*epsdJ
            modp = ising.p(newhJ)
            return np.nansum((log2p-np.log2(modp))*p) / epsdJ**2
        
        hess = np.zeros((len(dJ),len(dJ)))
        if (not n_cpus is None) and n_cpus<=1:
            for i in range(len(dJ)):
                hess[i,i] = diag(i)
            for i,j in combinations(range(len(dJ)),2):
                hess[i,j] = hess[j,i] = off_diag((i,j))
        else:
            hess[np.eye(len(dJ))==1] = self.pool.map(diag, range(len(dJ)))
            hess[np.triu_indices_from(hess,k=1)] = self.pool.map(off_diag, combinations(range(len(dJ)),2))
            # subtract off linear terms to get Hessian (and not just cross derivative)
            hess[np.triu_indices_from(hess,k=1)] -= np.array([hess[i,i]/2+hess[j,j]/2
                                                            for i,j in combinations(range(len(dJ)),2)])
            # fill in lower triangle
            hess += hess.T
            hess[np.eye(len(dJ))==1] /= 2

        if check_stability:
            hess2 = self.dkl_curvature(epsdJ=epsdJ/2, check_stability=False, hJ=hJ, dJ=dJ)
            err = hess2 - hess
            if (np.abs(err/hess) > rtol).any():
                normerr = np.linalg.norm(err)
                errflag = 1
                msg = ("Finite difference estimate has not converged with rtol=%f. "+
                       "May want to shrink epsdJ. Norm error %f.")
                print(msg%(rtol,normerr))
            else:
                errflag = 0
                normerr = None
        else:
            errflag = None
            normerr = None

        if not full_output:
            return hess
        return hess, errflag, normerr
    
    @staticmethod
    def p2pk(p, allStates):
        """Convert the full probability distribution to the probability of having k votes
        in the majority. Assuming that n is odd.

        Parameters
        ----------
        p : ndarray

        Returns
        -------
        ndarray
            p(k)
        """
        
        n = allStates.shape[1]
        pk = np.zeros(n//2+1)
        kVotes = np.abs( allStates.sum(1) )
        for i in range(pk.size):
            pk[i] = p[kVotes==(i*2+1)].sum()
        return pk

    def maj_curvature(self, *args, **kwargs):
        """Wrapper for _dkl_curvature() to find best finite diff step size."""

        if not 'epsdJ' in kwargs.keys():
            kwargs['epsdJ'] = 1e-4
        if not 'check_stability' in kwargs.keys():
            kwargs['check_stability'] = True
        if 'full_output' in kwargs.keys():
            full_output = kwargs['full_output']
        else:
            full_output = False
        kwargs['full_output'] = True
        epsDecreaseFactor = 10
        
        converged = False
        prevHess, errflag, prevNormerr = self._maj_curvature(*args, **kwargs)
        kwargs['epsdJ'] /= epsDecreaseFactor
        while (not converged) and errflag:
            hess, errflag, normerr = self._maj_curvature(*args, **kwargs)
            # end loop if error starts increasing again
            if errflag and normerr<prevNormerr:
                prevHess = hess
                prevNormerr = normerr
                kwargs['epsdJ'] /= epsDecreaseFactor
            else:
                converged = True
        if not converged and not errflag:
            normerr = None
        hess = prevHess
        
        if full_output:
            return hess, errflag, normerr
        return hess

    def _maj_curvature(self,
                      hJ=None,
                      dJ=None,
                      epsdJ=1e-4,
                      n_cpus=None,
                      check_stability=False,
                      rtol=1e-3,
                      full_output=False):
        """Calculate the hessian of the KL divergence (Fisher information metric) w.r.t.
        the theta_{ij} parameters replacing the spin i by sampling from j for the number
        of k votes in the majority.

        Use single step finite difference method to estimate Hessian.
        
        Parameters
        ----------
        hJ : ndarray, None
            Ising model parameters.
        dJ : ndarray, None
            Linear perturbations in parameter space corresponding to Hessian at given hJ.
            These can be calculuated using self.solve_linearized_perturbation().
        epsdJ : float, 1e-4
            Step size for taking linear perturbation wrt parameters.
        n_cpus : int, None
        check_stability : bool, False
        rtol : float, 1e-3
            Relative tolerance for each entry in Hessian when checking stability.
        full_output : bool, False
            
        Returns
        -------
        ndarray
            Hessian.
        int (optional)
            Error flag. 1 indicates rtol was exceeded. None indicates that no check was
            done.
        float (optional)
            Norm difference between hessian with step size eps and eps/2.
        """
        
        n = self.n
        if hJ is None:
            hJ = self.hJ
            p = self.p2pk(self.p, self.allStates)
        else:
            p = self.p2pk(self.ising.p(hJ), self.allStates)
        log2p = np.log2(p)
        if dJ is None:
            dJ = self.dJ
        
        # diagonal entries
        def diag(i, hJ=hJ, ising=self.ising, dJ=dJ, p=p, p2pk=self.p2pk, allStates=self.allStates):
            newhJ = hJ.copy()
            newhJ += dJ[i]*epsdJ
            modp = p2pk(ising.p(newhJ), allStates)
            return (2*(log2p-np.log2(modp)).dot(p)) / epsdJ**2
            
        # Compute off-diagonal entries. These don't account for the subtraction of the
        # diagonal elements which are removed later To see this, expand D(theta_i+del,
        # theta_j+del) to second order.
        def off_diag(args, hJ=hJ, ising=self.ising, p2pk=self.p2pk, dJ=dJ, p=p, allStates=self.allStates):
            i, j = args
            newhJ = hJ.copy()
            newhJ += (dJ[i]+dJ[j])*epsdJ
            modp = p2pk(ising.p(newhJ), allStates)
            return (log2p-np.log2(modp)).dot(p) / epsdJ**2
        
        hess = np.zeros((len(dJ),len(dJ)))
        if (not n_cpus is None) and n_cpus<=1:
            for i in range(len(dJ)):
                hess[i,i] = diag(i)
            for i,j in combinations(range(len(dJ)),2):
                hess[i,j] = hess[j,i] = off_diag((i,j))
        else:
            hess[np.eye(len(dJ))==1] = self.pool.map(diag, range(len(dJ)))
            hess[np.triu_indices_from(hess,k=1)] = self.pool.map(off_diag, combinations(range(len(dJ)),2))
            # subtract off linear terms to get Hessian (and not just cross derivative)
            hess[np.triu_indices_from(hess,k=1)] -= np.array([hess[i,i]/2+hess[j,j]/2
                                                            for i,j in combinations(range(len(dJ)),2)])
            # fill in lower triangle
            hess += hess.T
            hess[np.eye(len(dJ))==1] /= 2

        if check_stability:
            hess2 = self.maj_curvature(epsdJ=epsdJ/2, check_stability=False, hJ=hJ, dJ=dJ)
            err = hess2 - hess
            if (np.abs(err/hess) > rtol).any():
                normerr = np.linalg.norm(err)
                errflag = 1
                msg = ("Finite difference estimate has not converged with rtol=%f. "+
                       "May want to shrink epsdJ. Norm error %f.")
                print(msg%(rtol,normerr))
            else:
                errflag = 0
                normerr = None
        else:
            errflag = None
            normerr = None

        if not full_output:
            return hess
        return hess, errflag, normerr

    def hess_eig(self, hess, orientation_vector=None, imag_norm_threshold=1e-10):
        """Get Hessian eigenvalues and eigenvectors corresponds to parameter combinations
        of max curvature. Return them nicely sorted and cleaned and oriented consistently.
        
        Parameters
        ----------
        hess : ndarray
        orientation_vector : ndarray, None
            Vector along which to orient all vectors so that they are consistent with
            sign. By default, it is set to the sign of the first entry in the vector.
        imag_norm_threshold : float, 1e-10
        
        Returns
        -------
        ndarray
            Eigenvalues.
        ndarray
            Eigenvectors in cols.
        """
        
        if orientation_vector is None:
            orientation_vector = np.zeros(len(self.dJ))
            orientation_vector[0] = 1.

        eigval, eigvec = np.linalg.eig(hess)
        if (np.linalg.norm(eigval.imag)>imag_norm_threshold or
            np.linalg.norm(eigvec.imag[:,:10]>imag_norm_threshold)):
            print("Imaginary components are significant.")
        eigval = eigval.real
        eigvec = eigvec.real

        # orient all vectors along same direction
        eigvec *= np.sign(eigvec.T.dot(orientation_vector))[None,:]
        
        # sort by largest eigenvalues
        sortix = np.argsort(np.abs(eigval))[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
        # orient along direction of mean of individual means change
        eigvec *= np.sign(eigvec[:self.n,:].mean(0))[None,:]
        if (eigval<0).any():
            print("Negative eigenvalues.")
            print(eigval)
            print()
        
        return eigval, eigvec

    def hess_eig2dJ(self, eigvec, dJ=None):
        if dJ is None:
            dJ = self.dJ
        return dJ.T.dot(eigvec)

    def map_trajectory(self, n_steps, step_size, eigix=0, hJ0=None, initial_direction_sign=1):
        """Move along steepest directions of parameter step and keep a record of local
        landscape.
        
        Parameters
        ----------
        n_steps : int
        step_size : float
            Amount to move in specified direction accounting for the curvature. In other
            words, the distance moved, eps, will be step_size / eigval[eigix], such that
            steps are smaller in steeper regions.
        eigix : int, 0
            eigenvector direction in which to move. Default specifies principal direction.
        hJ0 : ndarray, None
        initial_direction_sign : int, 1
            -1 or 1.

        Returns
        -------
        dJ, hess, eigval, eigvec, hJTraj
        """

        if hJ0 is None:
            hJ0 = self.hJ

        dJ = []
        hess = []
        eigval = []
        eigvec = []
        hJTraj = [hJ0]
        flipRecord = np.ones(n_steps)
        prevStepFlipped = False

        for i in range(n_steps):
            p = self.ising.p(hJTraj[i])
            sisj = self.ising.calc_observables(hJTraj[i])

            dJ.append(np.zeros_like(self.dJ))
            for iStar in range(self.n):
                dJ[i][iStar], errflag = self.solve_linearized_perturbation(iStar, p=p, sisj=sisj)

            hess.append( self.dkl_curvature( hJ=hJTraj[i], dJ=dJ[i], epsdJ=1e-5) )
            out = self.hess_eig(hess[i])
            eigval.append(out[0])
            eigvec.append(out[1])
            
            # take a step in the steepest direction while moving in the same direction as the previous step
            #moveDirection = eigvec[i][:,eigix]
            # weighted average direction
            moveDirection = eigvec[i].dot(eigval[i]/np.linalg.norm(eigval[i]))
            dJcombo = self.hess_eig2dJ(moveDirection, dJ[-1])
            if i==0 and initial_direction_sign==-1:
                dJcombo *= -1
                flipRecord[0] = -1
                prevStepFlipped = True
            elif i>0:
                if prevStepFlipped:
                    if (prevMoveDirection.dot(moveDirection)<=0):
                        prevStepFlipped = False
                    else:
                        dJcombo *= -1
                        flipRecord[i] = -1
                        prevStepFlipped = True
                else:
                    if (prevMoveDirection.dot(moveDirection)<=0):
                        dJcombo *= -1
                        flipRecord[i] = -1
                        prevStepFlipped = True
                    else:
                        prevStepFlipped = False
            prevMoveDirection = moveDirection
                
            #hJTraj.append(hJTraj[-1] + dJcombo*step_size/eigval[-1][eigix])
            hJTraj.append(hJTraj[-1] + dJcombo*step_size/eigval[-1].sum())
            print("Done with step %d."%i)
        
        # apply sign change to return eigenvector direction
        for i in range(n_steps):
            #eigvec[i][:,eigix] *= flipRecord[i]
            eigvec[i] *= flipRecord[i]
        return dJ, hess, eigval, eigvec, hJTraj

    def find_peak_dkl_curvature(self, hJ0=None):
        """Use scipy.optimize to find peak in DKL curvature. Regions of parameter space
        where the matrix A describing the linearized perturbations is badly conditioned
        will be ignored by the algorithm.

        Parameters
        ----------
        hJ0 : ndarray, None

        Returns
        -------
        scipy.optimize.minimize dict
        """

        from scipy.optimize import minimize

        if hJ0 is None:
            hJ0 = self.hJ

        def f(hJ):
            p = self.ising.p(hJ)
            sisj = self.ising.calc_observables(hJ)
            dJ = np.zeros_like(self.dJ)
            for i in range(self.n):
                dJ[i], errflag = self.solve_linearized_perturbation(i,
                                                                    p=p,
                                                                    sisj=sisj,
                                                                    check_stability=False)
                if errflag:
                    return np.inf
            try:
                hessEigSum = np.linalg.eig(self.dkl_curvature(hJ=hJ, dJ=dJ))[0].sum()
            except np.linalg.LinAlgError:
                print("Problem with finding Hessian.")
                print(hJ)
                return np.inf
            return -hessEigSum

        return minimize(f, hJ0, options={'eps':1e-5, 'ftol':1e-4}, bounds=[(-1,1)]*len(hJ0))

    def __get_state__(self):
        return {'n':self.n,
                'h':self.hJ[:self.n],
                'J':self.hJ[self.n:],
                'dJ':self.dJ,
                'eps':self.eps}

    def __set_state__(self, state_dict):
        self.__init__(state_dict['n'], state_dict['h'], state_dict['J'], state_dict['eps'],
                      precompute=False)
        self.dJ = state_dict['dJ']
#end IsingFisherCurvatureMethod1


class IsingFisherCurvatureMethod2(IsingFisherCurvatureMethod1):
    def compute_dJ(self, p=None, sisj=None):
        # precompute linear change to parameters for small perturbation
        dJ = np.zeros((self.n*(self.n-1), self.n+(self.n-1)*self.n//2))
        counter = 0
        for i in range(self.n):
            for a in np.delete(range(self.n),i):
                dJ[counter], errflag = self.solve_linearized_perturbation(i, a, p=p, sisj=sisj)
                counter += 1
        return dJ

    def observables_after_perturbation(self, i, a, eps=None, perturb_up=False):
        """Make spin index i more like spin a by eps. Perturb the corresponding mean and
        the correlations with other spins j.
        
        Parameters
        ----------
        i : int
            Spin being perturbed.
        a : int
            Spin to mimic.
        eps : float, None
        perturb_up : bool, False

        Returns
        -------
        ndarray
            Observables <si> and <sisj> after perturbation.
        """
        
        if not hasattr(i,'__len__'):
            i = (i,)
        if not hasattr(a,'__len__'):
            a = (a,)
        for (i_,a_) in zip(i,a):
            assert i_!=a_
        if not hasattr(eps,'__len__'):
            eps = eps or self.eps
            eps = [eps]*len(i)
        n = self.n
        si = self.sisj[:n]
        sisj = self.sisj[n:]

        # observables after perturbations
        siNew = si.copy()
        sisjNew = sisj.copy()
        
        if perturb_up:
            for i_,a_,eps_ in zip(i,a,eps):
                self._observables_after_perturbation_up(siNew, sisjNew, i_, a_, eps_)
        else:
            for i_,a_,eps_ in zip(i,a,eps):
                self._observables_after_perturbation_down(siNew, sisjNew, i_, a_, eps_)


        return np.concatenate((siNew, sisjNew))
   
    def _observables_after_perturbation_up(self, si, sisj, i, a, eps):
        n = self.n

        si[i] = (1-eps)*si[i] + eps*si[a]

        for j in np.delete(range(n),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)

            if j==a:
                sisj[ijix] = (1-eps)*sisj[ijix] + eps
            else:
                if j<a:
                    jaix = unravel_index((j,a),n)
                else:
                    jaix = unravel_index((a,j),n)
                sisj[ijix] = (1-eps)*sisj[ijix] + eps*sisj[jaix]

    def _observables_after_perturbation_down(self, si, sisj, i, a, eps):
        n = self.n

        si[i] = (1-eps)*si[i] - eps*si[a]

        for j in np.delete(range(n),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)

            if j==a:
                sisj[ijix] = (1-eps)*sisj[ijix] - eps
            else:
                if j<a:
                    jaix = unravel_index((j,a),n)
                else:
                    jaix = unravel_index((a,j),n)
                sisj[ijix] = (1-eps)*sisj[ijix] - eps*sisj[jaix]

    def _solve_linearized_perturbation(self, iStar, aStar):
        """Consider a perturbation to a single spin.
        
        Parameters
        ----------
        iStar : int
        full_output : bool, False

        Returns
        -------
        """
        
        n = self.n
        p = self.p
        C = self.observables_after_perturbation(iStar, aStar)
        
        from coniii.solvers import Enumerate
        solver = Enumerate(n, calc_observables_multipliers=self.ising.calc_observables)
        return (solver.solve(C)-self.hJ)/self.eps

    def solve_linearized_perturbation(self, iStar, aStar,
                                      p=None,
                                      sisj=None,
                                      full_output=False,
                                      eps=None,
                                      check_stability=True):
        """Consider a perturbation to a single spin.
        
        Parameters
        ----------
        iStar : int
        aStar : int
        p : ndarray, None
        sisj : ndarray, None
        full_output : bool, False
        eps : float, None
        check_stability : bool, False

        Returns
        -------
        ndarray
            dJ
        int
            Error flag. Returns 0 by default. 1 means badly conditioned matrix A.
        tuple (optional)
            (A,C)
        """
        
        eps = eps or self.eps
        n = self.n
        if p is None:
            p = self.p
        if sisj is None:
            si = self.sisj[:n]
            sisj = self.sisj[n:]
        else:
            si = sisj[:n]
            sisj = sisj[n:]
        A = np.zeros((n+n*(n-1)//2, n+n*(n-1)//2))
        C = self.observables_after_perturbation(iStar, aStar, eps=eps)
        
        # mean constraints
        for i in range(n):
            for k in range(n):
                if i==k:
                    A[i,i] = 1 - C[i]*si[i]
                else:
                    if i<k:
                        ikix = unravel_index((i,k),n)
                    else:
                        ikix = unravel_index((k,i),n)
                    A[i,k] = sisj[ikix] - C[i]*si[k]

            for klcount,(k,l) in enumerate(combinations(range(n),2)):
                A[i,n+klcount] = self.triplets[(i,k,l)].dot(p) - C[i]*sisj[klcount]
        
        # pair constraints
        for ijcount,(i,j) in enumerate(combinations(range(n),2)):
            for k in range(n):
                A[n+ijcount,k] = self.triplets[(i,j,k)].dot(p) - C[n+ijcount]*si[k]
            for klcount,(k,l) in enumerate(combinations(range(n),2)):
                A[n+ijcount,n+klcount] = self.quartets[(i,j,k,l)].dot(p) - C[n+ijcount]*sisj[klcount]
    
        C -= self.sisj
        # factor out linear dependence on eps
        dJ = np.linalg.solve(A,C)/eps

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps, errflag = self.solve_linearized_perturbation(iStar, aStar,
                                                                     p=p,
                                                                     sisj=np.concatenate((si,sisj)),
                                                                     eps=eps/2,
                                                                     check_stability=False)
            # print if relative change is more than .1% for any entry
            if ((np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ)))>-3).any():
                print("Unstable solution. Recommend shrinking eps.")

        if np.linalg.cond(A)>1e15:
            warn("A is badly conditioned.")
            errflag = 1
        else:
            errflag = 0

        if full_output:
            return dJ, errflag, (A, C)
        return dJ, errflag

    def map_trajectory(self, n_steps, step_size, hJ0=None):
        """Linear perturbations to parameter space to explore info space.
        
        Parameters
        ----------
        n_steps : int
        step_size : int
        hJ0 : ndarray, None

        Returns
        -------
        """
        if hJ0 is None:
            hJ0 = self.hJ

        dJ = []
        hess = []
        eigval = []
        eigvec = []
        hJTraj = [hJ0]

        for i in range(n_steps):
            p = self.ising.p(hJTraj[i])
            sisj = self.ising.calc_observables(hJTraj[i])
            dJ.append(np.zeros_like(self.dJ))
            
            for count,(iStar,aStar) in enumerate(combinations(range(self.n),2)):
                dJ[i][count], errflag = self.solve_linearized_perturbation(iStar, aStar, p=p, sisj=sisj)

            hess.append( self.dkl_curvature( hJ=hJTraj[i], dJ=dJ[i]) )
            out = self.hess_eig(hess[i])
            eigval.append(out[0])
            eigvec.append(out[1])
            
            # take a step in the steepest direction
            eigix = 0
            dJcombo = self.hess_eig2dJ(eigvec[i][:,eigix])
            
            hJTraj.append(hJTraj[-1] + dJcombo*step_size)

        return dJ, hess, eigval, eigvec, hJTraj

    def find_peak_dkl_curvature(self, hJ0=None):
        """Use scipy.optimize to find peak in DKL curvature. Regions of parameter space
        where the matrix A describing the linearized perturbations is badly conditioned
        will be ignored by the algorithm.

        Parameters
        ----------
        hJ0 : ndarray, None

        Returns
        -------
        scipy.optimize.minimize dict
        """

        from scipy.optimize import minimize

        if hJ0 is None:
            hJ0 = self.hJ

        def f(hJ):
            p = self.ising.p(hJ)
            sisj = self.ising.calc_observables(hJ)
            dJ = np.zeros_like(self.dJ)
            counter = 0
            for i in range(self.n):
                for a in np.delete(range(self.n),i):
                    dJ[counter], errflag = self.solve_linearized_perturbation(i, a,
                                                                              p=p,
                                                                              sisj=sisj,
                                                                              check_stability=False)
                    counter += 1
                if errflag:
                    return np.inf
            try:
                hessEigSum = np.linalg.eig(self.dkl_curvature(hJ=hJ, dJ=dJ))[0].sum()
            except np.linalg.LinAlgError:
                print("Problem with finding Hessian.")
                print(hJ)
                return np.inf
            return -hessEigSum

        return minimize(f, hJ0, options={'eps':1e-5, 'ftol':1e-4}, bounds=[(-1,1)]*len(hJ0))
#end IsingFisherCurvatureMethod2


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


