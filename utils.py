# ============================================================================================ #
# Quick access to useful modules from SCOTUS project.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ============================================================================================ # 
import numpy as np
from numba import njit
from coniii.utils import *
import importlib


class IsingFisherCurvatureMethod1():
    def __init__(self, n, h=None, J=None, eps=1e-7, precompute=True):
        """
        Parameters
        ----------
        n : int
        h : ndarray, None
        J : ndarray, None
        eps : float, 1e-7
        precompute : bool, True
        """

        assert n>1 and 0<eps<.1
        self.n = n
        self.eps = eps
        self.hJ = np.concatenate((h,J))

        self.ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        self.sisj = self.ising.calc_observables(self.hJ)
        self.p = self.ising.p(self.hJ)
        self.allStates = bin_states(n, True)

        # precompute linear change to parameters for small perturbation
        self.dJ = np.zeros((n,n+(n-1)*n//2))
        if precompute:
            for i in range(n):
                self.dJ[i] = self.solve_linearized_perturbation(i)

    def observables_after_perturbation(self, i,
                                       eps=None,
                                       perturb_up=True):
        """Perturb spin index i by forcing it point upwards with probability eps/2.
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

        n = self.n
        eps = eps or self.eps
        si = self.sisj[:n]
        sisj = self.sisj[n:]
        
        if perturb_up:
            # observables after perturbations
            siNew = si.copy()
            sisjNew = sisj.copy()
            siNew[i]  = (1-eps)*si[i] + eps

            for j in np.delete(range(n),i):
                if i<j:
                    ijix = unravel_index((i,j),n)
                else:
                    ijix = unravel_index((j,i),n)
                sisjNew[ijix] = (1-eps)*sisj[ijix] + eps*si[j]
        else:
            # observables after perturbations
            siNew = si.copy()
            sisjNew = sisj.copy()
            siNew[i]  = (1-eps)*si[i] - eps

            for j in np.delete(range(n),i):
                if i<j:
                    ijix = unravel_index((i,j),n)
                else:
                    ijix = unravel_index((j,i),n)
                sisjNew[ijix] = (1-eps)*sisj[ijix] - eps*si[j]
        return np.concatenate((siNew, sisjNew))
    
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
            Linear change in maxent parameters.
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
        dJ[:self.n] *= -1
        return dJ

    def solve_linearized_perturbation(self, iStar,
                                      p=None,
                                      sisj=None,
                                      full_output=False,
                                      eps=None,
                                      check_stability=True):
        """Consider a perturbation to a single spin.
        
        Parameters
        ----------
        iStar : int
        p : ndarray, None
        sisj : ndarray, None
        full_output : bool, False
        eps : float, None
        check_stability : bool, False

        Returns
        -------
        ndarray
            dJ
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
                A[i,n+klcount] = np.prod(self.allStates[:,(i,k,l)],1).dot(p) - C[i]*sisj[klcount]
        
        # pair constraints
        for ijcount,(i,j) in enumerate(combinations(range(n),2)):
            for k in range(n):
                A[n+ijcount,k] = np.prod(self.allStates[:,(i,j,k)],1).dot(p) - C[n+ijcount]*si[k]
            for klcount,(k,l) in enumerate(combinations(range(n),2)):
                A[n+ijcount,n+klcount] = np.prod(self.allStates[:,(i,j,k,l)],1).dot(p) - C[n+ijcount]*sisj[klcount]
    
        C -= self.sisj
        # factor out linear dependence on eps
        dJ = np.linalg.solve(A,C)/eps
        if si[iStar]>=0:
            dJ[:n] *= -1

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps = self.solve_linearized_perturbation(iStar,
                                                            eps=eps/2,
                                                            check_stability=False,
                                                            p=p,
                                                            sisj=np.concatenate((si,sisj)))
            # print if relative change is more than .1% for any entry
            if ((np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ)))>-3).any():
                print("Unstable solution. Recommend shrinking eps.")
                    
        if full_output:
            return dJ, (A, C)
        return dJ

    def dkl_curvature(self,
                      hJ=None,
                      dJ=None,
                      epsdJ=1e-4,
                      n_cpus=None,
                      check_stability=False,
                      rtol=1e-3):
        """Calculate the hessian of the KL divergence (Fisher information metric) w.r.t.
        the theta_{ij} parameters replacing the spin i by sampling from j.
        
        Parameters
        ----------
        hJ : ndarray, None
            Ising model parameters.
        dJ : ndarray, None
            Linear perturbations in parameter space corresponding to Hessian at given hJ. These can be
            calculuated using self.solve_linearized_perturbation().
        epsdJ : float, 1e-2
            Step size for taking linear perturbation wrt parameters.
        n_cpus : int, None
        check_stability : bool, False
        rtol : float, 1e-3
            Relative tolerance for each entry in Hessian when checking stability.
            
        Returns
        -------
        ndarray
            Hessian.
        """
        
        from multiprocess import Pool, cpu_count
        from itertools import combinations
        
        n_cpus = n_cpus or (cpu_count()-1)
        n = self.n
        if hJ is None:
            hJ = self.hJ
            log2p = np.log2(self.p)
        else:
            log2p = np.log2(self.ising.p(hJ))
        if dJ is None:
            dJ = self.dJ
        
        # diagonal entries
        def diag(i, hJ=hJ, ising=self.ising, dJ=dJ):
            newhJ = hJ.copy()
            newhJ += 2*dJ[i]*epsdJ
            modp2 = ising.p(newhJ)
            
            newhJ = hJ.copy()
            newhJ += dJ[i]*epsdJ
            modp1 = ising.p(newhJ)
            return ((np.log2(modp2)-log2p).dot(modp2) -
                    2*(np.log2(modp1)-log2p).dot(modp1))/epsdJ**2
            
        # compute off-diagonal entries
        def off_diag(args, hJ=hJ, ising=self.ising, dJ=dJ):
            i, j = args
            newhJ = hJ.copy()
            newhJ += (dJ[i]+dJ[j])*epsdJ
            modp11 = ising.p(newhJ)
            
            newhJ = hJ.copy()
            newhJ += dJ[i]*epsdJ
            modp10 = ising.p(newhJ)
            
            newhJ = hJ.copy()
            newhJ += dJ[j]*epsdJ
            modp01 = ising.p(newhJ)
                    
            return ( (np.log2(modp11)-log2p).dot(modp11) -
                     (np.log2(modp10)-log2p).dot(modp10) - 
                     (np.log2(modp01)-log2p).dot(modp01) )/epsdJ**2
        
        hess = np.zeros((len(dJ),len(dJ)))
        if n_cpus<=1:
            for i in range(len(dJ)):
                hess[i,i] = diag(i)
            for i,j in combinations(range(len(dJ)),2):
                hess[i,j] = off_diag((i,j))
        else:
            pool = Pool(n_cpus)
            hess[np.triu_indices_from(hess,k=1)] = pool.map(off_diag, combinations(range(len(dJ)),2))
            hess += hess.T
            hess[np.eye(len(dJ))==1] = pool.map(diag, range(len(dJ)))
            pool.close()

        if check_stability:
            hess2 = self.dkl_curvature(epsdJ=epsdJ/2, check_stability=False)
            err = hess2 - hess
            if ((np.abs(err)/hess)>rtol).any():
                normerr = np.linalg.norm(err)
                msg = "Finite difference estimate has not converged. May want to shrink epsdJ. %f"%normerr
                print(msg)
    #     Another way of calculating Hessian for checking
    #     for i,j in combinations(range(4),2):
    #         newhJ = hJ.copy()
    #         newhJ[n:] += (dJ[i]+dJ[j])*epsDkl
    #         modp1 = ising.p(newhJ)
    #         newhJ = hJ.copy()
    #         newhJ[n:] += dJ[i]*epsDkl
    #         modp0 = ising.p(newhJ)
    #         d1 = ((log2(modp1)-log2(p)).dot(modp1) - (log2(modp0)-log2(p)).dot(modp0)) / epsDkl
            
    #         newhJ = hJ.copy()
    #         newhJ[n:] += (dJ[j])*epsDkl
    #         modp1 = ising.p(newhJ)
    #         newhJ = hJ.copy()
    #         modp0 = ising.p(newhJ)
    #         d0 = ((log2(modp1)-log2(p)).dot(modp1)-(log2(modp0)-log2(p)).dot(modp0)) / epsDkl

    #         hess[i,j] = hess[j,i] = (d1-d0)/epsDkl
        return hess

    def _dkl_curvature(self,
                       hJ=None,
                       dJ=None,
                       epsdJ=1e-4,
                       n_cpus=None,
                       check_stability=False,
                       rtol=1e-3):
        """Calculate the hessian of the KL divergence (Fisher information metric) w.r.t.
        the theta_{ij} parameters replacing the spin i by sampling from j.
        
        Parameters
        ----------
        hJ : ndarray, None
            Ising model parameters.
        dJ : ndarray, None
            Linear perturbations in parameter space corresponding to Hessian at given hJ. These can be
            calculuated using self.solve_linearized_perturbation().
        epsdJ : float, 1e-2
            Step size for taking linear perturbation wrt parameters.
        n_cpus : int, None
        check_stability : bool, False
        rtol : float, 1e-3
            Relative tolerance for each entry in Hessian when checking stability.
            
        Returns
        -------
        ndarray
            Hessian.
        """
        
        from multiprocess import Pool, cpu_count
        from itertools import combinations
        
        n_cpus = n_cpus or (cpu_count()-1)
        n = self.n
        allStatesSum = self.allStates.sum(1)
        kAllStatesSum = np.unique(allStatesSum)
        def pk(p):
            pk = np.zeros(kAllStatesSum.size)
            for i,k in enumerate(kAllStatesSum):
                pk[i] = p[allStatesSum==k].sum()
            return pk

        if hJ is None:
            hJ = self.hJ
            log2p = np.log2(pk(self.p))
        else:
            log2p = np.log2(pk(self.ising.p(hJ)))
        if dJ is None:
            dJ = self.dJ
        
        # diagonal entries
        def diag(i, hJ=hJ, ising=self.ising, dJ=dJ):
            newhJ = hJ.copy()
            newhJ += 2*dJ[i]*epsdJ
            modp2 = pk(ising.p(newhJ))
            
            newhJ = hJ.copy()
            newhJ += dJ[i]*epsdJ
            modp1 = pk(ising.p(newhJ))
            return ((np.log2(modp2)-log2p).dot(modp2) -
                    2*(np.log2(modp1)-log2p).dot(modp1))/epsdJ**2
            
        # compute off-diagonal entries
        def off_diag(args, hJ=hJ, ising=self.ising, dJ=dJ):
            i, j = args
            newhJ = hJ.copy()
            newhJ += (dJ[i]+dJ[j])*epsdJ
            modp11 = pk(ising.p(newhJ))
            
            newhJ = hJ.copy()
            newhJ += dJ[i]*epsdJ
            modp10 = pk(ising.p(newhJ))
            
            newhJ = hJ.copy()
            newhJ += dJ[j]*epsdJ
            modp01 = pk(ising.p(newhJ))
                    
            return ( (np.log2(modp11)-log2p).dot(modp11) -
                     (np.log2(modp10)-log2p).dot(modp10) - 
                     (np.log2(modp01)-log2p).dot(modp01) )/epsdJ**2
        
        hess = np.zeros((len(dJ),len(dJ)))
        if n_cpus<=1:
            for i in range(len(dJ)):
                hess[i,i] = diag(i)
            for i,j in combinations(range(len(dJ)),2):
                hess[i,j] = off_diag((i,j))
        else:
            pool = Pool(n_cpus)
            hess[np.triu_indices_from(hess,k=1)] = pool.map(off_diag, combinations(range(len(dJ)),2))
            hess += hess.T
            hess[np.eye(len(dJ))==1] = pool.map(diag, range(len(dJ)))
            pool.close()

        if check_stability:
            hess2 = self._dkl_curvature(epsdJ=epsdJ/2, check_stability=False)
            err = hess2 - hess
            if ((np.abs(err)/hess)>rtol).any():
                normerr = np.linalg.norm(err)
                msg = "Finite difference estimate has not converged. May want to shrink epsdJ. %f"%normerr
                print(msg)
        return hess

    def hess_eig(self, hess, imag_norm_threshold=1e-10):
        """Get Hessian eigenvalues and eigenvectors corresponds to parameter combinations
        of max curvature. Return them nicely sorted and cleaned.
        
        Parameters
        ----------
        hess : ndarray
        imag_norm_threshold : float, 1e-10
        
        Returns
        -------
        ndarray
            Eigenvalues.
        ndarray
            Eigenvectors in cols.
        """
        
        eigval, eigvec = np.linalg.eig(hess)
        if (np.linalg.norm(eigval.imag)>imag_norm_threshold or
            np.linalg.norm(eigvec.imag[:,:10]>imag_norm_threshold)):
            print("Imaginary components are significant.")
        eigval = eigval.real
        eigvec = eigvec.real
        
        # sort by largest eigenvalues
        sortix = np.argsort(np.abs(eigval))[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
        eigvec *= np.sign(eigvec.mean(0))[None,:]
        if (eigval<0).any():
            print("Negative eigenvalues.")
            print(eigval)
            print()
        
        return eigval, eigvec

    def hess_eig2dJ(self, eigvec):
        return self.dJ.T.dot(eigvec)

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
            
            for iStar in range(self.n):
                dJ[i][iStar] = self.solve_linearized_perturbation(iStar, p=p, sisj=sisj)

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
        from scipy.optimize import minimize

        if hJ0 is None:
            hJ0 = self.hJ

        def f(hJ):
            p = self.ising.p(hJ)
            sisj = self.ising.calc_observables(hJ)
            dJ = np.zeros_like(self.dJ)
            for i in range(self.n):
                dJ[i] = self.solve_linearized_perturbation(i, p=p, sisj=sisj, check_stability=False)
            hessDiagSum = self.dkl_curvature(hJ=hJ, dJ=dJ).diagonal().sum()
            print(hessDiagSum)
            print(hJ)
            print()
            return -hessDiagSum
            return -np.linalg.norm(self.dkl_curvature(hJ=hJ, dJ=dJ))

        return minimize(f, hJ0, options={'eps':1e-4, 'ftol':1e-3})
#end IsingFisherCurvatureMethod1


class IsingFisherCurvatureMethod2(IsingFisherCurvatureMethod1):
    def __init__(self, n, h=None, J=None, eps=1e-7):
        """Turning spin i into spin a."""

        assert n>1 and 0<eps<.1
        self.n = n
        self.eps = eps
        self.hJ = np.concatenate((h,J))

        self.ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        self.sisj = self.ising.calc_observables(self.hJ)
        self.p = self.ising.p(self.hJ)
        self.allStates = bin_states(n,True)

        # precompute linear change to parameters for small perturbation
        self.dJ = np.zeros((n*(n-1),n+(n-1)*n//2))
        counter = 0
        for i in range(n):
            for a in np.delete(range(n),i):
                self.dJ[counter] = self.solve_linearized_perturbation(i, a)
                counter += 1

    def observables_after_perturbation(self, i, a, eps=None):
        """Make spin index i more like spin a by eps. Perturb the corresponding mean and
        the correlations with other spins j.
        
        Parameters
        ----------
        i : int
            Spin being perturbed.
        a : int
            Spin to mimic.
        eps : float, None

        Returns
        -------
        ndarray
            Observables <si> and <sisj> after perturbation.
        """
        
        assert i!=a
        n = self.n
        eps = eps or self.eps
        si = self.sisj[:n]
        sisj = self.sisj[n:]

        # observables after perturbations
        siNew = si.copy()
        sisjNew = sisj.copy()
        siNew[i]  = (1-eps)*si[i] + eps*si[a]

        for j in np.delete(range(n),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)

            if j==a:
                sisjNew[ijix] = (1-eps)*sisj[ijix] + eps
            else:
                if j<a:
                    jaix = unravel_index((j,a),n)
                else:
                    jaix = unravel_index((a,j),n)
                sisjNew[ijix] = (1-eps)*sisj[ijix] + eps*sisj[jaix]
        return np.concatenate((siNew, sisjNew))
    
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
                A[i,n+klcount] = np.prod(self.allStates[:,(i,k,l)],1).dot(p) - C[i]*sisj[klcount]
        
        # pair constraints
        for ijcount,(i,j) in enumerate(combinations(range(n),2)):
            for k in range(n):
                A[n+ijcount,k] = np.prod(self.allStates[:,(i,j,k)],1).dot(p) - C[n+ijcount]*si[k]
            for klcount,(k,l) in enumerate(combinations(range(n),2)):
                A[n+ijcount,n+klcount] = np.prod(self.allStates[:,(i,j,k,l)],1).dot(p) - C[n+ijcount]*sisj[klcount]
    
        C -= self.sisj
        # factor out linear dependence on eps
        dJ = np.linalg.solve(A,C)/eps

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps = self.solve_linearized_perturbation(iStar, aStar,
                                                            p=p,
                                                            sisj=np.concatenate((si,sisj)),
                                                            eps=eps/2,
                                                            check_stability=False)
            # print if relative change is more than .1% for any entry
            if ((np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ)))>-3).any():
                print("Unstable solution. Recommend shrinking eps.")
        if full_output:
            return dJ, (A, C)
        return dJ

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
                dJ[i][count] = self.solve_linearized_perturbation(iStar, aStar, p=p, sisj=sisj)

            hess.append( self.dkl_curvature( hJ=hJTraj[i], dJ=dJ[i]) )
            out = self.hess_eig(hess[i])
            eigval.append(out[0])
            eigvec.append(out[1])
            
            # take a step in the steepest direction
            eigix = 0
            dJcombo = self.hess_eig2dJ(eigvec[i][:,eigix])
            
            hJTraj.append(hJTraj[-1] + dJcombo*step_size)

        return dJ, hess, eigval, eigvec, hJTraj
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
