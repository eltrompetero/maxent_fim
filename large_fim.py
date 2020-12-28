# ====================================================================================== #
# Classes for calculating FIM on Ising and Potts models for large systems where MC
# sampling is necessary to calculate perturbative quantities.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *

from numba import njit, prange
from coniii.enumerate import fast_logsumexp, mp_fast_logsumexp
from multiprocess import Pool, cpu_count, set_start_method
from scipy.sparse import coo_matrix
from numba.typed import Dict as nDict
from tempfile import mkdtemp
from multiprocess import RawArray
import socket

from .models import LargeIsing, LargePotts3

np.seterr(divide='ignore')



class Magnetization():
    """Perturbation of local magnetizations one at a time. By default, perturbation is
    towards +1 and not -1.
    """
    def __init__(self, n,
                 h=None,
                 J=None,
                 eps=1e-7,
                 precompute=True,
                 n_cpus=None,
                 n_samples=100_000):
        """
        Parameters
        ----------
        n : int
        h : ndarray, None
        J : ndarray, None
        eps : float, 1e-7
        precompute : bool, True
        n_cpus : int, None
        n_samples : int, 10_000_000
            Number of samples for Metropolis sampling.
        """

        assert n>1 and 0<eps<.1
        self.n = n
        self.kStates = 2
        self.eps = eps
        self.hJ = np.concatenate((h,J))
        self.n_cpus = n_cpus

        self.ising = LargeIsing((h,J), n_samples)
        self.sisj = np.concatenate(self.ising.corr[:2])
        self.p = self.ising.p
        self.allStates = self.ising.states.astype(np.int8)
        _, self.coarseInvix = np.unique(np.abs(self.allStates.sum(1)), return_inverse=True)
        self.coarseUix = np.unique(self.coarseInvix)
        
        # cache triplet and quartet products
        self._triplets_and_quartets() 
    
        if precompute:
            self.dJ = self.compute_dJ()
        else:
            self.dJ = np.zeros((self.n,self.n+(self.n-1)*self.n//2))

        self._custom_end_init()

    def _custom_end_init(self):
        """Placeholder that can be replaced in children classes."""
        return
    
    def _triplets_and_quartets(self):
        self.triplets, self.quartets = jit_triplets_and_quartets(self.n, self.allStates.astype(np.int8)) 

    def compute_dJ(self, p=None, sisj=None):
        # precompute linear change to parameters for small perturbation
        dJ = np.zeros((self.n,self.n+(self.n-1)*self.n//2))
        for i in range(self.n):
            dJ[i], errflag = self.solve_linearized_perturbation(i, p=p, sisj=sisj)
        return dJ

    def observables_after_perturbation(self, i,
                                       eps=None):
        """Perturb all specified spin by forcing it point upwards with probability eps/2.
        Perturb the corresponding mean and the correlations with other spins j.
        
        Parameters
        ----------
        i : int
            Spin to perturb.
        eps : float, None

        Returns
        -------
        ndarray
            Observables <si> and <sisj> after perturbation.
        bool, True
            If True, made the specified spin point up +1. If False, made it point down -1.
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
        
        # observables after perturbations
        for i_,eps_ in zip(i,eps):
            jit_observables_after_perturbation_plus_field(n, siNew, sisjNew, i_, eps_)
        perturb_up = True

        return np.concatenate((siNew, sisjNew)), perturb_up
   
    def _observables_after_perturbation_plus_field(self, n, si, sisj, i, eps):
        """        
        Parameters
        ----------
        n : int
        si : ndarray
        sisj : ndarray
        i : int
        eps : float
        """

        # observables after perturbations
        si[i]  = (1-eps)*si[i] + eps

        for j in np.delete(list(range(n)),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)
            sisj[ijix] = (1-eps)*sisj[ijix] + eps*si[j]

    def _solve_linearized_perturbation_tester(self, iStar, eps=None):
        """Consider a perturbation to a single spin.
        
        Parameters
        ----------
        iStar : int
        eps : float, None
        perturb_up : bool, False

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
        C, perturb_up = self.observables_after_perturbation(iStar, eps=eps)

        solver = Enumerate(np.ones((1,n)))
        if perturb_up:
            return (solver.solve(constraints=C) - self.hJ) / eps

        # account for sign of perturbation on fields
        dJ = -(solver.solve(constraints=C) - self.hJ) / eps
        return dJ

    def solve_linearized_perturbation(self, *args, **kwargs):
        """Wrapper for automating search for best eps value for given perturbation.
        """
        
        # settings
        epsChangeFactor = 10
        
        # check whether error increases or decreases with eps
        eps0 = kwargs.get('eps', self.eps)
        kwargs['check_stability'] = True
        kwargs['full_output'] = True
        
        dJ, errflag, (A,C), relerr = self._solve_linearized_perturbation(*args, **kwargs)

        kwargs['eps'] = eps0*epsChangeFactor
        dJUp, errflagUp, _, relerrUp = self._solve_linearized_perturbation(*args, **kwargs)

        kwargs['eps'] = eps0/epsChangeFactor
        dJDown, errflagDown, _, relerrDown = self._solve_linearized_perturbation(*args, **kwargs)
        
        # if changing eps doesn't help, just return estimate at current eps
        if relerr.max()<relerrUp.max() and relerr.max()<relerrDown.max():
            return dJ, errflag
        
        # if error decreases more sharpy going down
        if relerrDown.max()<=relerrUp.max():
            epsChangeFactor = 1/epsChangeFactor
            prevdJ, errflag, prevRelErr = dJDown, errflagDown, relerrDown
        # if error decreases more sharpy going up, no need to change eps
        else:
            prevdJ, errflag, prevRelErr = dJUp, errflagUp, relerrUp
        
        # decrease/increase eps til error starts increasing
        converged = False
        while (not converged) and errflag:
            kwargs['eps'] *= epsChangeFactor
            dJ, errflag, (A,C), relerr = self._solve_linearized_perturbation(*args, **kwargs)
            if errflag and relerr.max()<prevRelErr.max():
                prevdJ = dJ
                prevRelErr = relerr
            else:
                converged = True
        
        return dJ, errflag

    def _solve_linearized_perturbation(self, iStar,
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
        
        perturb_up = False
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
        A = np.zeros((n+n*(n-1)//2, n+n*(n-1)//2), dtype=si.dtype)
        C, perturb_up = self.observables_after_perturbation(iStar, eps=eps)
        
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
            dJ = np.linalg.lstsq(A, C, rcond=None)[0]/eps
        # Since default is to perturb down
        if not perturb_up:
            dJ *= -1

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps, errflag = self._solve_linearized_perturbation(iStar,
                                                                      eps=eps/2,
                                                                      check_stability=False,
                                                                      p=p,
                                                                      sisj=np.concatenate((si,sisj)))
            # print if relative change is more than .1% for any entry
            relerr = np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ))
            if (relerr>-3).any():
                print("Unstable solution. Recommend shrinking eps. %E"%(10**relerr.max()))
        else:
            relerr = None
                   
        if (np.linalg.cond(A)>1e15):
            warn("A is badly conditioned.")
            errflag = 1
        else:
            errflag = 0
        if full_output:
            return dJ, errflag, (A, C), relerr
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
    def p2pk(p, uix, invix):
        """Convert the full probability distribution to the probability of having k votes
        in the majority.

        Parameters
        ----------
        p : ndarray
        uix : ndarray
        invix : ndarray

        Returns
        -------
        ndarray
            p(k)
        """
         
        pk = np.zeros(len(uix))
        for i in range(len(uix)):
            pk[i] = p[invix==i].sum()

        return pk

    @staticmethod
    def logp2pk(E, uix, invix):
        """Convert the full probability distribution to the probability of having k votes
        in the majority.

        Parameters
        ----------
        E : ndarray
            Energies of each configuration.
        uix : ndarray
        invix : ndarray

        Returns
        -------
        ndarray
            The unnormalized log probability: log p(k) + logZ.
        """
         
        logsumEk = np.zeros(len(uix))
        for i in range(len(uix)):
            logsumEk[i] = fast_logsumexp(-E[invix==i])[0]
        return logsumEk

    def maj_curvature(self, *args, **kwargs):
        """Wrapper for _maj_curvature() to find best finite diff step size.
        
        Returns
        -------
        ndarray
            Calculated Hessian.
        int
            Error flag.
        ndarray
            Errors.
        """

        import multiprocess as mp

        if not 'epsdJ' in kwargs.keys():
            kwargs['epsdJ'] = 1e-4
        if not 'check_stability' in kwargs.keys():
            kwargs['check_stability'] = True
        full_output = kwargs.get('full_output', False)
        if 'high_prec' in kwargs.keys():
            high_prec = kwargs['high_prec']
            del kwargs['high_prec']
        else:
            high_prec = False
        kwargs['full_output'] = True
        epsDecreaseFactor = 10
        
        # start loop for finding optimal eps for Hessian with num diff
        converged = False
        if high_prec:
            prevHess, errflag, preverr = self._maj_curvature_high_prec(*args, **kwargs)
        else:
            prevHess, errflag, preverr = self._maj_curvature(*args, **kwargs)
        kwargs['epsdJ'] /= epsDecreaseFactor
        while (not converged) and errflag:
            if high_prec:
                hess, errflag, err = self._maj_curvature_high_prec(*args, **kwargs)
            else:
                hess, errflag, err = self._maj_curvature(*args, **kwargs)
            # end loop if error starts increasing again
            if errflag and np.linalg.norm(err)<np.linalg.norm(preverr):
                prevHess = hess
                preverr = err
                kwargs['epsdJ'] /= epsDecreaseFactor
            else:
                converged = True

        hess = prevHess
        err = preverr
        
        if full_output:
            return hess, errflag, err
        return hess

    def _maj_curvature(self,
                       hJ=None,
                       dJ=None,
                       epsdJ=1e-7,
                       check_stability=False,
                       rtol=1e-3,
                       full_output=False,
                       calc_off_diag=True,
                       calc_diag=True,
                       iprint=True):
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
        check_stability : bool, False
        rtol : float, 1e-3
            Relative tolerance for each entry in Hessian when checking stability.
        full_output : bool, False
        calc_off_diag : bool, True
        calc_diag : bool, True
        iprint : bool, True
            
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
        E = calc_e(self.allStates, hJ)
        logZ = fast_logsumexp(-E)[0]
        logsumEk = self.logp2pk(E, self.coarseUix, self.coarseInvix)
        p = np.exp(logsumEk - logZ)
        assert np.isclose(p.sum(),1), p.sum()
        if dJ is None:
            dJ = self.dJ
            assert self.dJ.shape[1]==(n+n*(n-1)//2)
        if iprint:
            print('Done with preamble.')

        # diagonal entries of hessian
        def diag(i, hJ=hJ, dJ=dJ, p=self.p, pk=p, logp2pk=self.logp2pk,
                 uix=self.coarseUix, invix=self.coarseInvix,
                 n=self.n, E=E, logZ=logZ, allStates=self.allStates):
            # round eps step to machine precision
            mxix = np.abs(dJ[i]).argmax()
            newhJ = hJ[mxix] + dJ[i][mxix]*epsdJ  # tilde parameters
            epsdJ_ = (newhJ-hJ[mxix]) / dJ[i][mxix]  # eps
            if np.isnan(epsdJ_): return 0.
            # change in effective k majority energy is sum over micro states
            correction = calc_e(allStates, dJ[i]*epsdJ_)
            correction = np.array([correction[invix==ix].dot(p[invix==ix])/p[invix==ix].sum()
                                   for ix in range(uix.size)])
            num = ((correction.dot(pk) - correction)**2).dot(pk)
            dd = num / np.log(2) / epsdJ_**2
            if iprint and np.isnan(dd):
                print('nan for diag', i, epsdJ_)
            
            return dd

        # off-diagonal entries of hessian
        def off_diag(args, hJ=hJ, dJ=dJ, p=self.p, pk=p, logp2pk=self.logp2pk,
                     uix=self.coarseUix, invix=self.coarseInvix,
                     n=self.n, E=E, logZ=logZ, allStates=self.allStates):
            i, j = args
            
            # round eps step to machine precision
            mxix = np.abs(dJ[i]).argmax()
            newhJ = hJ[mxix] + dJ[i][mxix]*epsdJ
            epsdJi = (newhJ - hJ[mxix])/dJ[i][mxix]/2
            if np.isnan(epsdJi): return 0.
            correction = calc_e(allStates, dJ[i]*epsdJi)
            correctioni = np.array([correction[invix==ix].dot(p[invix==ix])/p[invix==ix].sum()
                                    for ix in range(uix.size)])

            # round eps step to machine precision
            mxix = np.abs(dJ[j]).argmax()
            newhJ = hJ[mxix] + dJ[j][mxix]*epsdJ
            epsdJj = (newhJ - hJ[mxix])/dJ[j][mxix]/2
            if np.isnan(epsdJj): return 0.
            correction = calc_e(allStates, dJ[j]*epsdJj)
            correctionj = np.array([correction[invix==ix].dot(p[invix==ix])/p[invix==ix].sum()
                                    for ix in range(uix.size)])

            num = ((correctioni.dot(pk) - correctioni)*(correctionj.dot(pk) - correctionj)).dot(pk)
            dd = num / np.log(2) / (epsdJi * epsdJj)
            if iprint and np.isnan(dd):
                print('nan for off diag', args, epsdJi, epsdJj)
            return dd
        
        hess = np.zeros((len(dJ),len(dJ)))
        if not 'pool' in self.__dict__.keys():
            warn("Not using multiprocess can lead to excessive memory usage.")
            if calc_diag:
                for i in range(len(dJ)):
                    hess[i,i] = diag(i)
                if iprint:
                    print("Done with diag.")
            if calc_off_diag:
                for i,j in combinations(range(len(dJ)),2):
                    hess[i,j] = off_diag((i,j))
                    if iprint:
                        print("Done with off diag (%d,%d)."%(i,j))
                if iprint:
                    print("Done with off diag.")
        else:
            if calc_diag:
                hess[np.eye(len(dJ))==1] = self.pool.map(diag, range(len(dJ)))
                if iprint:
                    print("Done with diag.")
            if calc_off_diag:
                hess[np.triu_indices_from(hess,k=1)] = self.pool.map(off_diag, combinations(range(len(dJ)),2))
                if iprint:
                    print("Done with off diag.")

        if calc_off_diag:
            # fill in lower triangle
            hess += hess.T
            hess[np.eye(len(dJ))==1] /= 2

        # check for precision problems
        assert ~np.isnan(hess).any(), hess
        assert ~np.isinf(hess).any(), hess

        if check_stability:
            hess2 = self._maj_curvature(epsdJ=epsdJ/2,
                                        check_stability=False,
                                        iprint=iprint,
                                        hJ=hJ,
                                        dJ=dJ,
                                        calc_diag=calc_diag,
                                        calc_off_diag=calc_off_diag)
            err = hess - hess2
            if (np.abs(err/hess) > rtol).any():
                errflag = 1
                if iprint:
                    msg = ("Finite difference estimate has not converged with rtol=%f. "+
                           "May want to shrink epsdJ. Norm error %f.")
                    print(msg%(rtol,np.linalg.norm(err)))
            else:
                errflag = 0
                if iprint:
                    msg = "Finite difference estimate converged with rtol=%f."
                    print(msg%rtol)
        else:
            errflag = None
            err = None

        if not full_output:
            return hess
        return hess, errflag, err

    def _maj_curvature_high_prec(self,
                                 hJ=None,
                                 dJ=None,
                                 epsdJ=1e-4,
                                 n_cpus=None,
                                 check_stability=False,
                                 rtol=1e-3,
                                 full_output=False,
                                 dps=20):
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
        dps : int, 20
            
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
        
        import mpmath as mp
        mp.mp.dps = dps

        mplog2_ = lambda x:mp.log(x)/mp.log(2)
        mplog2 = lambda x: list(map(mplog2_, x))
        n = self.n
        if hJ is None:
            hJ = self.hJ
        p = self.p2pk_high_prec(self.ising.p(hJ), self.coarseUix, self.coarseInvix)
        log2p = np.array(mplog2(p))
        if dJ is None:
            dJ = self.dJ

        def diag(i,
                 hJ=hJ,
                 ising=self.ising,
                 dJ=dJ,
                 p=p,
                 p2pk=self.p2pk_high_prec,
                 uix=self.coarseUix,
                 invix=self.coarseInvix):
            # round epsdJ_ to machine precision
            mxix = np.argmax(np.abs(dJ[i]))
            newhJ = hJ[mxix] + dJ[i][mxix]*epsdJ
            epsdJ_ = (newhJ - hJ[mxix]) / dJ[i][mxix] / 2
            if np.isnan(epsdJ_): return 0.

            newhJ = hJ + dJ[i]*epsdJ_
            modp = p2pk(ising.p(newhJ), uix, invix)
            dklplus = 2*(log2p-mplog2(modp)).dot(p)

            newhJ -= 2*dJ[i]*epsdJ_
            modp = p2pk(ising.p(newhJ), uix, invix)
            dklminus = 2*(log2p-mplog2(modp)).dot(p)

            return (dklplus+dklminus) / 2 / epsdJ_**2

        # theta_j+del) to second order.
        def off_diag(args,
                     hJ=hJ,
                     ising=self.ising,
                     p2pk=self.p2pk_high_prec,
                     dJ=dJ,
                     p=p,
                     uix=self.coarseUix,
                     invix=self.coarseInvix):
            i, j = args
            
            # round epsdJ_ to machine precision
            mxix = np.argmax(np.abs(dJ[i]+dJ[j]))
            newhJ = hJ[mxix] + (dJ[i][mxix]+dJ[j][mxix])*epsdJ
            epsdJ_ = (newhJ - hJ[mxix]) / (dJ[i][mxix]+dJ[j][mxix]) / 2
            if np.isnan(epsdJ_): return 0.

            newhJ = hJ + (dJ[i]+dJ[j])*epsdJ_
            modp = p2pk(ising.p(newhJ), uix, invix)
            dklplus = (log2p-mplog2(modp)).dot(p)

            newhJ -= 2*(dJ[i]+dJ[j])*epsdJ_
            modp = p2pk(ising.p(newhJ), uix, invix)
            dklminus = (log2p-mplog2(modp)).dot(p)
            
            return (dklplus+dklminus) / 2 / epsdJ_**2

        hess = np.zeros((len(dJ),len(dJ)))
        if (not n_cpus is None) and n_cpus<=1:
            for i in range(len(dJ)):
                hess[i,i] = diag(i)
            for i,j in combinations(range(len(dJ)),2):
                hess[i,j] = off_diag((i,j))
        else:
            hess[np.eye(len(dJ))==1] = self.pool.map(diag, range(len(dJ)))
            hess[np.triu_indices_from(hess,k=1)] = self.pool.map(off_diag, combinations(range(len(dJ)),2))

        # subtract off linear terms to get Hessian (and not just cross derivative)
        hess[np.triu_indices_from(hess,k=1)] -= np.array([hess[i,i]/2+hess[j,j]/2
                                                        for i,j in combinations(range(len(dJ)),2)])
        # fill in lower triangle
        hess += hess.T
        hess[np.eye(len(dJ))==1] /= 2
        
        assert ~np.isnan(hess).any()
        assert ~np.isinf(hess).any()

        if check_stability:
            hess2 = self._maj_curvature_high_prec(epsdJ=epsdJ/2,
                                                  check_stability=False,
                                                  hJ=hJ,
                                                  dJ=dJ,
                                                  n_cpus=n_cpus)
            err = (hess - hess2)*4/3
            if (np.abs(err/hess) > rtol).any():
                errflag = 1
                msg = ("Finite difference estimate has not converged with rtol=%f. "+
                       "May want to shrink epsdJ. Norm error %f.")
                print(msg%(rtol,np.linalg.norm(err)))
            else:
                errflag = 0
                msg = "Finite difference estimate converged with rtol=%f."
                print(msg%rtol)
        else:
            errflag = None
            err = None

        if not full_output:
            return hess
        return hess, errflag, err

    def hess_eig(self, hess,
                 orientation_vector=None,
                 imag_norm_threshold=1e-10,
                 iprint=True):
        """Get Hessian eigenvalues and eigenvectors corresponds to parameter combinations
        of max curvature. Return them nicely sorted and cleaned and oriented consistently.
        
        Parameters
        ----------
        hess : ndarray
        orientation_vector : ndarray, None
            Vector along which to orient all vectors so that they are consistent with
            sign. By default, it is set to the sign of the first entry in the vector.
        imag_norm_threshold : float, 1e-10
        iprint : bool, True
        
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
        sortix = np.argsort(eigval)[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
        # orient along direction of mean of individual means change
        eigvec *= np.sign(eigvec[:self.n,:].mean(0))[None,:]
        if iprint and (eigval<0).any():
            print("There are negative eigenvalues.")
            print()
        
        return eigval, eigvec

    def hess_eig2dJ(self, eigvec, dJ=None):
        if dJ is None:
            dJ = self.dJ
        return dJ.T.dot(eigvec)

    def dlogpk(self, ds, eps=1e-4, ds_as_dJ=False):
        """Rate of change in log[p(k)] when moving along perturbation direction described
        by given vector.

        Parameters
        ----------
        ds : ndarray
            Perturbation specification that will be mapped to change in underlying maxent
            parameters, e.g. a col of the eigenmatrix.
        eps : float, 1e-4
        ds_as_dJ : bool, False
            If True, simple perturbation of maxent model parameters is given.

        Returns
        -------
        list of ndarray
            Each vector specifies rate of change in p(k) ordered where the number of
            voters in the majority decreases by one voter at a time.
        float
            Norm difference between gradient calculation with eps step and 2*eps step.
        """

        # map change in correlation space to that in amxent parameter space
        if ds_as_dJ:
            dJ = ds
        else:
            dJ = self.dJ.T.dot(ds)

        dlogp = self._dlogpk(dJ, eps)
        # check what derivative looks like for larger eps to check convergence
        dlogpCoarse = self._dlogpk(dJ, eps * 2)

        return dlogp, np.linalg.norm(dlogpCoarse - dlogp)
        
    def _dlogpk(self, dJ, eps):
        from coniii.utils import define_ising_helper_functions
        calc_e = define_ising_helper_functions()[0]

        # calculate change in energy of each observed configuration as we induce
        # perturbation in both pos and neg directions
        dE = calc_e(self.allStates, dJ*eps)
        E = np.log(self.p)
        pplus = np.exp(E+dE - fast_logsumexp(E+dE)[0])  # modified probability distribution
        pminus = np.exp(E-dE - fast_logsumexp(E-dE)[0])  # modified probability distribution

        pkplusdE = np.zeros(n//2+1)
        pkminusdE = np.zeros(n//2+1)
        for k in range(n//2+1):
            pkplusdE[k] = pplus[np.abs(self.allStates.sum(1))==(n-k*2)].sum()
            pkminusdE[k] = pminus[np.abs(self.allStates.sum(1))==(n-k*2)].sum()
        dlogp = (np.log2(pkplusdE) - np.log2(pkminusdE)) / (2*eps)
        return dlogp

    def component_subspace_dlogpk(self, hess, eps=1e-5):
        """Rate of change in log[p(k)] when moving along the principal mode of each
        component's subspace.

        See "2019-08-01 detail about CAA 99's affect on p(k).ipynb"

        Parameters
        ----------
        hess : ndarray
        eps : float, 1e-5

        Returns
        -------
        list of ndarray
            Each vector specifies rate of change in p(k) ordered where the number of
            voters in the majority decreases by one voter at a time.
        """
        
        from .spectral import block_subspace_eig
        from coniii.utils import define_ising_helper_functions
        calc_e, calc_observables, _ = define_ising_helper_functions()
        n = self.n
        dlogp = []
        
        for ix in range(n):
            # iterate over components whose subspaces we explore
            # subspace eigenvector
            eigval, eigvec = block_subspace_eig(hess, n-1)

            v = eigvec[ix][:,0].real  # take principal eigenvector
            dE = calc_e(self.allStates.astype(np.int64), v.dot(self.dJ[ix*(n-1):(ix+1)*(n-1)])/(n-1))*eps
            E = np.log(self.p)
            pplus = np.exp(E+dE - fast_logsumexp(E+dE)[0])  # modified probability distribution
            pminus = np.exp(E-dE - fast_logsumexp(E-dE)[0])  # modified probability distribution

            pkplusdE = np.zeros(n//2+1)
            pkminusdE = np.zeros(n//2+1)
            for k in range(n//2+1):
                pkplusdE[k] = pplus[np.abs(self.allStates.sum(1))==(n-k*2)].sum()
                pkminusdE[k] = pminus[np.abs(self.allStates.sum(1))==(n-k*2)].sum()
            dlogp.append( (np.log2(pkplusdE) - np.log2(pkminusdE))/(2*eps) )
        return dlogp

    def __get_state__(self):
        # always close multiprocess pool when pickling
        if 'pool' in self.__dict__.keys():
            self.pool.close()
            del self.pool

        return {'n':self.n,
                'h':self.hJ[:self.n],
                'J':self.hJ[self.n:],
                'dJ':self.dJ,
                'eps':self.eps,
                'n_cpus':self.n_cpus}

    def __set_state__(self, state_dict):
        self.__init__(state_dict['n'], state_dict['h'], state_dict['J'], state_dict['eps'],
                      precompute=False,
                      n_cpus=state_dict.get('n_cpus',None))
        self.dJ = state_dict['dJ']
#end Magnetization



class MagnetizationConstant(Magnetization):
    """Perturbation of local magnetizations one at a time keeping fixed the amount of
    perturbation (this is akin to replacing only states that are contrary to the objective
    direction.
    """
    def observables_after_perturbation(self, i, eps=None):
        """Perturb all specified spin by forcing its magnetization by eps.
        
        Parameters
        ----------
        i : int
        eps : float, None

        Returns
        -------
        ndarray
            Observables <si> and <sisj> after perturbation.
        bool
            If True, made the specified spin point up +1. If False, made it point down -1.
        """
        
        if not hasattr(i,'__len__'):
            i = (i,)
        if not hasattr(eps,'__len__'):
            eps = eps or self.eps
            eps = [eps]*len(i)
        n = self.n
        si = self.sisj[:n]
        sisj = self.sisj[n:]
        
        # try perturbing up first
        siNew = si.copy()
        sisjNew = sisj.copy()
        perturb_up = True
        for i_,eps_ in zip(i,eps):
            # observables after perturbations
            jit_observables_after_perturbation_plus_mean(n, siNew, sisjNew, i_, eps_)
        # if we've surpassed the allowed values for correlations then try perturbing down
        # there is no check to make sure this perturbation doesn't lead to impossible values
        #if (np.abs(siNew)>1).any() or (np.abs(sisjNew)>1).any():
        #    siNew = si.copy()
        #    sisjNew = sisj.copy()
        #    perturb_up = False
        #    for i_,eps_ in zip(i,eps):
        #        # observables after perturbations
        #        jit_observables_after_perturbation_minus_mean(n, siNew, sisjNew, i_, eps_)

        return np.concatenate((siNew, sisjNew)), perturb_up
   
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
        C, perturb_up = self.observables_after_perturbation(iStar, eps=eps)
        
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
        # Since default is to perturb down
        if not perturb_up:
            dJ *= -1

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps, errflag = self.solve_linearized_perturbation(iStar,
                                                                     eps=eps/2,
                                                                     check_stability=False,
                                                                     p=p,
                                                                     sisj=np.concatenate((si,sisj)))
            # print if relative change is more than .1% for any entry
            relerr = np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ))
            if (relerr>-3).any():
                print("Unstable solution. Recommend shrinking eps. %E"%(10**relerr.max()))
                   
        if np.linalg.cond(A)>1e15:
            warn("A is badly conditioned.")
            errflag = 1
        else:
            errflag = 0
        if full_output:
            return dJ, errflag, (A, C)
        return dJ, errflag
#end Magnetizationa


class Coupling(Magnetization):
    """Perturbation that increases correlation between pairs of spins.
    """
    def compute_dJ(self, p=None, sisj=None):
        # precompute linear change to parameters for small perturbation
        dJ = np.zeros((self.n*(self.n-1), self.n+(self.n-1)*self.n//2))
        counter = 0
        for i in range(self.n):
            for a in np.delete(range(self.n),i):
                dJ[counter], errflag = self.solve_linearized_perturbation(i, a, p=p, sisj=sisj)
                counter += 1
        return dJ

    @staticmethod
    def _observables_after_perturbation_up(si, sisj, i, a, eps):
        n = len(si)

        si[i] = 1 - eps*(si[i] - si[a])

        for j in delete(list(range(n)),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)

            if j==a:
                sisj[ijix] = 1 - eps*(sisj[ijix] - 1)
            else:
                if j<a:
                    jaix = unravel_index((j,a),n)
                else:
                    jaix = unravel_index((a,j),n)
                sisj[ijix] = 1 - eps*(sisj[ijix] - sisj[jaix])
    
    @staticmethod
    def _observables_after_perturbation_down(si, sisj, i, a, eps):
        n = len(si)

        si[i] = 1 - eps*(si[i] + si[a])

        for j in delete(list(range(n)),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)

            if j==a:
                sisj[ijix] = 1 - eps*(sisj[ijix] + 1)
            else:
                if j<a:
                    jaix = unravel_index((j,a),n)
                else:
                    jaix = unravel_index((a,j),n)
                sisj[ijix] = 1 - eps*(sisj[ijix] + sisj[jaix])

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
        
        for i_,a_,eps_ in zip(i,a,eps):
            jit_observables_after_perturbation_plus(n, siNew, sisjNew, i_, a_, eps_)

        return np.concatenate((siNew, sisjNew))
    
    def _solve_linearized_perturbation_tester(self, iStar, aStar):
        """
        ***FOR DEBUGGING ONLY***
        
        Consider a perturbation to a single spin.
        
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
        solver = Enumerate(np.ones((1,n)))
        return (solver.solve(constraints=C)-self.hJ)/self.eps
    
    def solve_linearized_perturbation(self, *args, **kwargs):
        """Wrapper for automating search for best eps value for given perturbation.
        """
        
        # settings
        epsChangeFactor = 10
        
        # check whether error increases or decreases with eps
        eps0 = kwargs.get('eps', self.eps)
        kwargs['check_stability'] = True
        kwargs['full_output'] = True
        
        dJ, errflag, (A,C), relerr = self._solve_linearized_perturbation(*args, **kwargs)

        kwargs['eps'] = eps0*epsChangeFactor
        dJUp, errflagUp, _, relerrUp = self._solve_linearized_perturbation(*args, **kwargs)

        kwargs['eps'] = eps0/epsChangeFactor
        dJDown, errflagDown, _, relerrDown = self._solve_linearized_perturbation(*args, **kwargs)
        
        # if changing eps doesn't help, just return estimate at current eps
        if relerr.max()<relerrUp.max() and relerr.max()<relerrDown.max():
            return dJ, errflag
        
        # if error decreases more sharpy going down
        if relerrDown.max()<=relerrUp.max():
            epsChangeFactor = 1/epsChangeFactor
            prevdJ, errflag, prevRelErr = dJDown, errflagDown, relerrDown
        # if error decreases more sharpy going up, no need to change eps
        else:
            prevdJ, errflag, prevRelErr = dJUp, errflagUp, relerrUp
        
        # decrease/increase eps til error starts increasing
        converged = False
        while (not converged) and errflag:
            kwargs['eps'] *= epsChangeFactor
            dJ, errflag, (A,C), relerr = self._solve_linearized_perturbation(*args, **kwargs)
            if errflag and relerr.max()<prevRelErr.max():
                prevdJ = dJ
                prevRelErr = relerr
            else:
                converged = True
        
        return dJ, errflag

    def _solve_linearized_perturbation(self, iStar, aStar,
                                      p=None,
                                      sisj=None,
                                      full_output=False,
                                      eps=None,
                                      check_stability=True,
                                      disp=False):
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
        float (optional)
            Relative error to log10.
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
        errflag = 0
        
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
            dJtwiceEps, errflag = self._solve_linearized_perturbation(iStar, aStar,
                                                                      p=p,
                                                                      sisj=np.concatenate((si,sisj)),
                                                                      eps=eps/2,
                                                                      check_stability=False)
            # print if relative change is more than .1% for any entry
            relerr = np.log10(np.abs(dJ-dJtwiceEps))-np.log10(np.abs(dJ))
            if (relerr>-3).any():
                if disp:
                    print("Unstable solution. Recommend shrinking eps. Max err=%E"%(10**relerr.max()))
                errflag = 2
        
        if np.linalg.cond(A)>1e15:
            warn("A is badly conditioned.")
            # this takes precedence over relerr over threshold
            errflag = 1

        if full_output:
            if check_stability:
                return dJ, errflag, (A, C), relerr
            return dJ, errflag, (A, C)
        return dJ, errflag
#end Couplings



class Mag3(Coupling):
    """Single spin mean perturbations for ternary states."""
    def __init__(self, n,
                 h=None,
                 J=None,
                 eps=1e-7,
                 precompute=True,
                 n_cpus=None,
                 n_samples=100_000,
                 rng=None,
                 iprint=True,
                 sampler_kw={}):
        """
        Parameters
        ----------
        n : int
        h : ndarray, None
            Full specification of all 3xN fields.
        J : ndarray, None
        eps : float, 1e-7
            Must be careful to set this relative to the precision of the MC sample.
        precompute : bool, True
            Set up and do perturation calculation.
        n_cpus : int, None
        n_samples : int, 100_000
            Number of samples for Metropolis sampling.
        rng : np.random.RandomState, None
        iprint : bool, True
            Display info if True.
        sampler_kw : dict, {}
        """

        assert isinstance(n, int) and n>1 and 0<eps<1e-2
        assert (h[2*n:3*n]==0).all()
        assert h.size==3*n and J.size==n*(n-1)//2

        self.n = n
        self.kStates = 3
        self.eps = eps
        self.hJ = np.concatenate((h,J))
        self.n_cpus = n_cpus
        self.rng = rng or np.random
        self.iprint = iprint

        self.ising = LargePotts3((h,J), n_samples, iprint=iprint, rng=self.rng, **sampler_kw)
        self.sisj = np.concatenate(self.ising.corr[:2])
        self.p = self.ising.p
        self.allStates = self.ising.states.astype(np.int8)
        # determine p(k) as the number of votes in the plurality
        kVotes = list(map(lambda x:np.sort(np.bincount(x, minlength=3))[::-1],
                          self.allStates))
        self.coarseUix, self.coarseInvix = np.unique(kVotes, return_inverse=True, axis=0)
        self.coarseUix = np.unique(self.coarseInvix)

        if precompute:
            # cache triplet and quartet products
            if self.iprint: print("Starting correlations calculation...")
            self._triplets_and_quartets() 
            if self.iprint: print("Done.")

            if iprint: print("Computing dJ...")
            self.compute_dJ()
            if iprint: print("Done.")
        else:
            self.dJ = None
    
    def __get_state__(self):
        """Throw away numba dict objects. These will have to be calculated again.
        """

        state = self.__dict__.copy()
        del state['pairs'], state['triplets'], state['quartets']

        return state

    def __set_state__(self, state):
        """Numba typed dicts need to be computed again."""
        self.__dict__.update(state) 
        # self._triplets_and_quartets()

    def _triplets_and_quartets(self):
        """Caching triplets and quartets for large systems is too memory intensive. Only
        pairwise are cached."""

        n = self.n
        kStates = self.kStates
        allStates = self.allStates

        pairs, triplets, quartets = jit_triplets_and_quartets(n, kStates, allStates, self.p)
        # copy these into normal python dicts
        self.pairs = dict(pairs)
        self.triplets = dict(triplets)
        self.quartets = dict(quartets)

    def compute_dJ(self, n_cpus=None):
        """Compute linear change to maxent parameters for perturbation.

        For each spin, push into each of k possible states such that perturbations are
        grouped into triplets.
        
        Parameters
        ----------
        n_cpus : int, None

        Returns
        -------
        dJ : ndarray
            (n_perturbation_parameters, n_maxent_parameters)
        """
        
        n_cpus = n_cpus or self.n_cpus

        def wrapper(params):
            """Push spin i towards state k"""
            from numba import typed, types

            i, k = params

            # convert dicts to numba typed dicts for calc_A
            pairs = typed.Dict.empty(types.UniTuple(types.int64, 4), types.float64)
            for ix, v in self.pairs.items():
                pairs[ix] = v
            triplets = typed.Dict.empty(types.UniTuple(types.int64, 4), types.float64)
            for ix, v in self.triplets.items():
                triplets[ix] = v
            quartets = typed.Dict.empty(types.UniTuple(types.int64, 4), types.float64)
            for ix, v in self.quartets.items():
                quartets[ix] = v
            self.pairs, self.triplets, self.quartets = pairs, triplets, quartets

            return self.solve_linearized_perturbation(i, k)[0]
        
        # run parallelized approach to solving perturbations
        if n_cpus is None or n_cpus>1:
            if self.iprint: print("Multiprocessing to solve for dJ...")
            with threadpool_limits(limits=1, user_api='blas'):
                with Pool(n_cpus) as pool:
                    dJ = np.vstack(list( pool.map(wrapper,
                                                  product(range(self.n), range(self.kStates))) ))
        else:
            dJ = np.zeros((self.n*self.kStates, self.kStates*self.n+(self.n-1)*self.n//2))
            for counter, (i, k) in enumerate(product(range(self.n), range(self.kStates))):
                dJ[counter] = wrapper((i, k))

        self.dJ = dJ
        return dJ
    
    def observables_after_perturbation(self, i, k, eps=None):
        """Push spin i towards state k by eps. Perturb the corresponding mean and the
        correlations with other spins j.
        
        Parameters
        ----------
        i : int
            Spin being perturbed.
        k : int
        eps : float, None
            To set custom epsilon parameter.

        Returns
        -------
        ndarray
            Perturbed observables <si> and <sisj> concatenated.
        """

        from .utils import perturb_3_spin
        
        eps = eps or self.eps
        n = self.n
        si = self.sisj[:n*3].copy()
        sisj = self.sisj[3*n:].copy()

        delta = perturb_3_spin(si[[i,i+n,i+2*n]], k, eps, return_delta=True)
        # modify this spin's bias
        si[[i,i+n,i+2*n]] += delta

        # change pairwise correlations with spin i
        for j in np.delete(list(range(n)), i):
            if i < j:
                ijix = unravel_index((i,j), n)
            else:
                ijix = unravel_index((j,i), n)
            
            for k_ in range(self.kStates):
                sisj[ijix] += delta[k_] * self.sisj[k_ * n + j]
        
        return np.concatenate((si, sisj))
   
    def _maj_curvature(self,
                       epsdJ=1e-7,
                       check_stability=False,
                       rtol=1e-3,
                       full_output=False,
                       calc_off_diag=True,
                       off_diag_ix=None,
                       calc_diag=True,
                       iprint=True):
        """Calculate the hessian of the KL divergence (Fisher information metric) w.r.t.
        the theta_{ij} parameters replacing the spin i by sampling from j for the number
        of k votes in the majority.

        Use single step finite difference method to estimate Hessian.

        Memory map is used to store results during computation. Shared memory is used to
        reduce time spent serializing parameters.
        
        Parameters
        ----------
        epsdJ : float, 1e-4
            Step size for taking linear perturbation wrt parameters.
        check_stability : bool, False
        rtol : float, 1e-3
            Relative tolerance for each entry in Hessian when checking stability.
        full_output : bool, False
        calc_off_diag : bool, True
        off_diag_ix : list of tuples, None
            If list of tuples, dictates which elements of off-diagonal to calculate. Since
            the matrix must be symmetric, no combinations of (i,j) indices can be repeated
            and i<j.
        calc_diag : bool, True
        iprint : bool, True
            
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
        E = calc_all_energies(n, self.kStates, self.allStates, self.hJ)
        logZ = fast_logsumexp(-E)[0]
        logsumEk = self.logp2pk(E, self.coarseUix, self.coarseInvix)
        # check if calc_off_diag specifies calculating all entries or just specific ones
        if off_diag_ix:
            assert all([i<j for i,j in off_diag_ix])
            assert np.unique(off_diag_ix, axis=0).shape[0]==len(off_diag_ix)
                
        # set up multiprocessing
        # shared memory on drive for FIM temporary results with exceptions for servers
        # that don't work well with this
        if 'wheeler' in socket.gethostname():
            mmfname = '%s/hess.dat'%mkdtemp(dir='/wheeler/scratch/edlee/')
        else:
            mmfname = '%s/hess.dat'%mkdtemp()
        mmhess = np.memmap(mmfname,
                           dtype=np.float64,
                           shape=(len(self.dJ),len(self.dJ)),
                           mode='w+')
        mmhess[:,:] = np.nan  # default value to make it easy to check saved results

        # shared memory for large arrays, these must be global to be accessed via inheritance
        global rdJ, rpk, rp, rAllStates, rcoarseUix, rcoarseInvix
        rdJ = RawArray('d', self.dJ.size)
        dJ = np.frombuffer(rdJ).reshape(self.dJ.shape)
        rpk = RawArray('d', logsumEk.size)
        pk = np.frombuffer(rpk)
        rp = RawArray('d', self.p.size)
        p = np.frombuffer(rp)
        rAllStates = RawArray('b', self.allStates.size)
        allStates = np.frombuffer(rAllStates, dtype=np.int8).reshape(self.allStates.shape)
        rcoarseUix = RawArray('l', self.coarseUix.size)
        coarseUix = np.frombuffer(rcoarseUix, dtype=np.int64)
        rcoarseInvix = RawArray('l', self.coarseInvix.size)
        coarseInvix = np.frombuffer(rcoarseInvix, dtype=np.int64)
        
        # fill in shared memory arrays
        np.copyto(dJ, self.dJ)
        np.copyto(pk, np.exp(logsumEk - logZ))
        np.copyto(p, self.p)
        np.copyto(allStates, self.allStates)
        np.copyto(coarseUix, self.coarseUix)
        np.copyto(coarseInvix, self.coarseInvix)

        shapesDict = {'dJ':dJ.shape,
                      'allStates':allStates.shape}  # necessary for reading in from mem

        assert np.isclose(pk.sum(),1), pk.sum()
        if iprint:
            print('Done with preamble.')
        
        # calculation
        # diagonal entries of hessian
        def diag(i, hJ=self.hJ, logp2pk=self.logp2pk,
                 n=self.n, logZ=logZ, kStates=self.kStates,
                 shapesDict=shapesDict):
            # load arrays from shared memory
            dJ = np.frombuffer(rdJ).reshape(shapesDict['dJ'])
            pk = np.frombuffer(rpk)
            p = np.frombuffer(rp)
            allStates = np.frombuffer(rAllStates, dtype=np.int8).reshape(shapesDict['allStates'])
            uix = np.frombuffer(rcoarseUix, dtype=np.int64)
            invix = np.frombuffer(rcoarseInvix, dtype=np.int64)
            
            # check for degeneracies
            if not np.isfinite(dJ[i][0]):
                dd = 0.
            else:
                # round eps step to machine precision
                mxix = np.abs(dJ[i]).argmax()
                newhJ = hJ[mxix] + dJ[i][mxix]*epsdJ
                epsdJ_ = (newhJ-hJ[mxix]) / dJ[i][mxix]
                if np.isnan(epsdJ_): return 0.
                correction = calc_all_energies(n, kStates, allStates, dJ[i]*epsdJ_)
                correction = np.array([correction[invix==ix].dot(p[invix==ix])/p[invix==ix].sum()
                                       for ix in range(len(uix))])
                num = ((correction.dot(pk) - correction)**2).dot(pk)
                dd = num / np.log(2) / epsdJ_**2
                if iprint and np.isnan(dd):
                    print('nan for diag', i, epsdJ_)
            
            # write result to memmap
            mmhessEntry = np.memmap(mmfname,
                                    dtype=np.float64,
                                    mode='r+',
                                    shape=(1,),
                                    offset=(i * len(dJ) + i) * 8)
            mmhessEntry[0] = dd
            del mmhessEntry

            return dd
         
        # off-diagonal entries of hessian
        def off_diag(args, hJ=self.hJ, logp2pk=self.logp2pk,
                     n=self.n, logZ=logZ, kStates=self.kStates,
                     shapesDict=shapesDict):
            i, j = args

            # load arrays from shared memory
            dJ = np.frombuffer(rdJ).reshape(shapesDict['dJ'])
            pk = np.frombuffer(rpk)
            p = np.frombuffer(rp)
            allStates = np.frombuffer(rAllStates, dtype=np.int8).reshape(shapesDict['allStates'])
            uix = np.frombuffer(rcoarseUix, dtype=np.int64)
            invix = np.frombuffer(rcoarseInvix, dtype=np.int64)
            
            # check for degenerate entries
            if not (np.isfinite(dJ[i][0]) and np.isfinite(dJ[j][0])):
                dd = 0.
            else:
                # round eps step to machine precision
                mxix = np.abs(dJ[i]).argmax()
                newhJ = hJ[mxix] + dJ[i][mxix]*epsdJ
                epsdJi = (newhJ - hJ[mxix])/dJ[i][mxix]/2
                if np.isnan(epsdJi): return 0.
                correction = calc_all_energies(n, kStates, allStates, dJ[i]*epsdJi)
                correctioni = np.array([correction[invix==ix].dot(p[invix==ix])/p[invix==ix].sum()
                                        for ix in range(len(uix))])

                # round eps step to machine precision
                mxix = np.abs(dJ[j]).argmax()
                newhJ = hJ[mxix] + dJ[j][mxix]*epsdJ
                epsdJj = (newhJ - hJ[mxix])/dJ[j][mxix]/2
                if np.isnan(epsdJj): return 0.
                correction = calc_all_energies(n, kStates, allStates, dJ[j]*epsdJj)
                correctionj = np.array([correction[invix==ix].dot(p[invix==ix])/p[invix==ix].sum()
                                        for ix in range(len(uix))])

                num = ((correctioni.dot(pk) - correctioni)*(correctionj.dot(pk) - correctionj)).dot(pk)
                dd = num / np.log(2) / (epsdJi * epsdJj)
                if iprint and np.isnan(dd):
                    print('nan for off diag', args, epsdJi, epsdJj)
            
            # write result to memmap
            mmhessEntry = np.memmap(mmfname,
                                    dtype=np.float64,
                                    mode='r+',
                                    shape=(1,),
                                    offset=(i * len(dJ) + j) * 8)
            mmhessEntry[0] = dd
            del mmhessEntry

            return dd
        
        hess = np.zeros((len(dJ),len(dJ)))
        
        if self.n_cpus is None or self.n_cpus>1:
            n_cpus = self.n_cpus or cpu_count()
            with threadpool_limits(limits=1, user_api='blas'):
                with Pool(n_cpus, maxtasksperchild=1) as pool:
                    if calc_diag:
                        hess[np.eye(len(dJ))==1] = pool.map(diag, range(len(dJ)))
                        if iprint:
                            print("Done with diag.")
                    if calc_off_diag:
                        if off_diag_ix:
                            hess[tuple(zip(*off_diag_ix))] = pool.map(off_diag, off_diag_ix)
                        else:
                            hess[np.triu_indices_from(hess,k=1)] = pool.map(off_diag,
                                                                            combinations(range(len(dJ)), 2))
                        if iprint:
                            print("Done with off diag.")
        else:
            warn("Not using multiprocess can lead to excessive memory usage.")
            if calc_diag:
                for i in range(len(dJ)):
                    hess[i,i] = diag(i)
                if iprint:
                    print("Done with diag.")
            if calc_off_diag:
                for i,j in off_diag_ix:
                    hess[i,j] = off_diag((i,j))
                    if iprint:
                        print("Done with off diag (%d,%d)."%(i,j))
                if iprint:
                    print("Done with off diag.")

        if calc_off_diag:
            # fill in lower triangle
            hess += hess.T
            hess[np.eye(len(dJ))==1] /= 2

        if check_stability:
            if iprint: print("Checking stability...")
            hess2 = self._maj_curvature(epsdJ=epsdJ/2,
                                        check_stability=False,
                                        iprint=iprint,
                                        calc_diag=calc_diag,
                                        calc_off_diag=calc_off_diag,
                                        off_diag_ix=off_diag_ix)
            # check stability for entries that have not been set to np.nan (either on
            # purpose or because of precision problems)
            nanix = ~(np.isnan(hess) | np.isnan(hess2))
            err = hess[nanix] - hess2[nanix]
            if (np.abs(err/hess[nanix]) > rtol).any():
                errflag = 1
                if iprint:
                    msg = ("Finite difference estimate has not converged with rtol=%f. "+
                           "May want to shrink epsdJ. Norm error %f.")
                    print(msg%(rtol,np.linalg.norm(err)))
            else:
                errflag = 0
                if iprint:
                    msg = "Finite difference estimate converged with rtol=%f."
                    print(msg%rtol)
        else:
            errflag = None
            err = None

        # get rid of memory map
        del mmhess  
        
        # return
        if not full_output:
            return hess
        return hess, errflag, err

    def _test_maj_curvature(self):
        n = self.n
        hJ = self.hJ
        E = calc_all_energies(n, self.kStates, hJ)
        logZ = fast_logsumexp(-E)[0]
        logsumEk = self.logp2pk(E, self.coarseUix, self.coarseInvix)
        p = np.exp(logsumEk - logZ)
        dJ = self.dJ

        # diagonal entries of hessian
        def diag(i, eps, hJ=hJ, dJ=dJ, p=p, logp2pk=self.logp2pk,
                 uix=self.coarseUix, invix=self.coarseInvix,
                 n=self.n, E=E, logZ=logZ, kStates=self.kStates):
            # round eps step to machine precision
            mxix = np.abs(dJ[i]).argmax()
            newhJ = hJ[mxix] + dJ[i][mxix]*eps
            eps = (newhJ-hJ[mxix]) / dJ[i][mxix]
            if np.isnan(eps): return 0.
            correction = calc_all_energies(n, kStates, dJ[i]*eps)
            
            # forward step
            Enew = E+correction
            modlogsumEkplus = logp2pk(Enew, uix, invix)
            #Zkplus = fast_logsumexp(-Enew)[0]
            
            # backwards step
            Enew = E-correction
            modlogsumEkminus = logp2pk(Enew, uix, invix)
            #Zkminus = fast_logsumexp(-Enew)[0]

            num = (logsumEk - modlogsumEkplus)**2
            ddplus = num.dot(p) / np.log(2) / eps**2

            num = (logsumEk - modlogsumEkminus)**2
            ddminus = num.dot(p) / np.log(2) / eps**2

            #num_ = 2*(logsumEk - logZ) + (Zkplus - modlogsumEkplus) + (Zkminus - modlogsumEkminus)
            #print( num_.dot(p) / np.log(2) / 2 / eps**2 )
            return ddplus, ddplus-ddminus
        return diag
  
    def _solve_linearized_perturbation_tester(self, iStar, kStar, full_output=False):
        """
        ***FOR DEBUGGING ONLY***

        Parameters
        ----------
        iStar : int
        kStar : int

        Returns
        -------
        ndarray
            Estimated linear change in maxent parameters.
        """
        
        n = self.n
        k = self.kStates
        p = self.p
        C = self.observables_after_perturbation(iStar, kStar)[0]
        assert k==3, "Only handles k=3."

        from coniii.solvers import Enumerate
        from coniii.models import Potts3
        model = Potts3([np.zeros(k*n), np.zeros(n*(n-1)//2)])
        calc_observables = define_ternary_helper_functions()[1]
        solver = Enumerate(np.vstack((np.ones(n), -np.ones(n))),
                           model=model,
                           calc_observables=calc_observables)
       
        # hybr solver seems to work more consistently than default krylov
        hJ0 = solver.solve(constraints=self.sisj,
                           initial_guess=self.hJ,
                           full_output=True,
                           scipy_solver_kwargs={'method':'hybr', 'tol':1e-12})[0]

        # hybr solver seems to work more consistently than default krylov
        fullsoln = solver.solve(constraints=C,
                                initial_guess=self.hJ,
                                full_output=True,
                                scipy_solver_kwargs={'method':'hybr', 'tol':1e-12})
        soln = fullsoln[0]

        # remove translational offset for last set of fields
        hJ0[:n*k] -= np.tile(hJ0[n*(k-1):n*k], k)
        soln[:n*k] -= np.tile(soln[n*(k-1):n*k], k)
        if full_output:
            return (soln - self.hJ)/(self.eps), fullsoln
        return (soln - hJ0)/(self.eps)

    def _solve_linearized_perturbation(self, iStar, kStar,
                                       full_output=False,
                                       eps=None,
                                       check_stability=True):
        """Consider a perturbation to match spin iStar to state kStar.
        
        Parameters
        ----------
        iStar : int
            Index of spin.
        kStar : int
            Configuration into which to fix spin iStar with small probability.
        full_output : bool, False
        eps : float, None
        check_stability : bool, True

        Returns
        -------
        ndarray
            dJ
        int
            Error flag. Returns 0 by default. 1 means badly conditioned matrix A. 2 means
            that the solution was unstable and probably untrustworthy.
        tuple (optional)
            (Aplus, Cplus)
        float (optional)
            Relative error to log10.
        """
        
        eps = eps or self.eps
        n = self.n
        kStates = self.kStates
        p = self.p
        si = self.sisj[:n*kStates]
        sisj = self.sisj[kStates*n:]

        # check if spin iStar is perfect magnetized, in which case dJ should be infinite
        tol = 2 / self.ising.sample_size
        if si[iStar]>=(1-tol) or si[iStar+n]>=(1-tol) or si[iStar+2*n]>=(1-tol):
            dJ = np.zeros(si.size+sisj.size) + np.inf
            errflag = 0
            relerr = np.zeros(1)

            if full_output:
                if check_stability:
                    return dJ, errflag, (None, None), relerr
                return dJ, errflag, (None, None)
            return dJ, errflag

        # matrix that will be multiplied by the vector of correlation perturbations
        Cplus = self.observables_after_perturbation(iStar, kStar, eps=eps)
        Cminus = self.observables_after_perturbation(iStar, kStar, eps=-eps)
        errflag = 0
        
        if type(self.pairs) is dict:
            warn("Using slower version of calc_A.")
            Aplus = calc_A(n, kStates,
                           self.allStates, p, si, sisj,
                           self.pairs, self.triplets, self.quartets,
                           Cplus)
            Aminus = calc_A(n, kStates,
                            self.allStates, p, si, sisj,
                            self.pairs, self.triplets, self.quartets,
                            Cminus)
        else:
            Aplus = jit_calc_A(n, kStates,
                               self.allStates, p, si, sisj,
                               self.pairs, self.triplets, self.quartets,
                               Cplus)
            Aminus = jit_calc_A(n, kStates,
                                self.allStates, p, si, sisj,
                                self.pairs, self.triplets, self.quartets,
                                Cminus)
        Cplus -= self.sisj
        Cminus -= self.sisj
        # factor out linear dependence on eps
        dJ = np.linalg.lstsq(Aplus+Aminus, Cplus-Cminus, rcond=None)[0]/eps
        # put back in fields that we've fixed to 0 by normalization
        dJ = np.insert(dJ, (kStates-1)*n, np.zeros(n))

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps, errflag = self._solve_linearized_perturbation(iStar, kStar,
                                                                      eps=eps/2,
                                                                      check_stability=False)
            # print if relative change is more than .1% for any entry excepting zeros
            # which are set by zeroed fields
            zeroix = dJ==0
            relerr = np.log10(np.abs(dJ[~zeroix]-dJtwiceEps[~zeroix])) - np.log10(np.abs(dJ[~zeroix]))
            if (relerr>-3).any():
                if self.iprint:
                    print("Unstable solution. Recommend shrinking eps. Max err=%E"%(10**relerr.max()))
                errflag = 2
        
        if np.linalg.cond(Aplus)>1e15:
            warn("A is badly conditioned for pair (i, a)=(%d, %d)."%(iStar,kStar))
            # this takes precedence over relerr over threshold
            errflag = 1

        if full_output:
            if check_stability:
                return dJ, errflag, (Aplus, Cplus), relerr
            return dJ, errflag, (Aplus, Cplus)
        return dJ, errflag

    def _dlogpk(self, dJ, eps):
        """Partitions are determined by unique ones found in the sample and referenced in
        self.coarseUix and self.coarseInvix.
        """

        calc_e = self.ising.calc_e
        n = self.n
        
        # calculate change in energy of each observed configuration as we induce
        # perturbation in both pos and neg directions
        dE = calc_e(self.allStates, dJ*eps)
        E = np.log(self.p)
        pplus = np.exp(E+dE - fast_logsumexp(E+dE)[0])  # modified probability distribution
        pminus = np.exp(E-dE - fast_logsumexp(E-dE)[0])  # modified probability distribution
        
        choosek = self.coarseUix.size
        pkplusdE = np.zeros(choosek)
        pkminusdE = np.zeros(choosek)
        for k in range(choosek):
            pkplusdE[k] = pplus[self.coarseInvix==k].sum()
            pkminusdE[k] = pminus[self.coarseInvix==k].sum()
        dlogp = (np.log2(pkplusdE) - np.log2(pkminusdE)) / (2*eps)
        return dlogp
#end Mag3



class Coupling3(Mag3):
    """Pairwise perturbations tweaked for ternary states."""
    def __init__(self, n,
                 h=None,
                 J=None,
                 eps=1e-7,
                 precompute=True,
                 n_cpus=None,
                 n_samples=100_000,
                 rng=None,
                 iprint=True,
                 sampler_kw={}):
        """
        Parameters
        ----------
        n : int
        h : ndarray, None
            Full specification of all 3xN fields.
        J : ndarray, None
        eps : float, 1e-7
            Must be careful to set this relative to the precision of the MC sample.
        precompute : bool, True
            Set up and do perturation calculation.
        n_cpus : int, None
        n_samples : int, 10_000_000
            Number of samples for Metropolis sampling.
        rng : np.random.RandomState, None
        iprint : bool, True
            Display info if True.
        sampler_kw : dict, {}
        """

        assert n>1 and 0<eps<1e-2
        assert (h[2*n:3*n]==0).all()
        assert h.size==3*n and J.size==n*(n-1)//2

        self.n = n
        self.kStates = 3
        self.eps = eps
        self.hJ = np.concatenate((h,J))
        self.n_cpus = n_cpus
        self.rng = rng or np.random
        self.iprint = iprint

        self.ising = LargePotts3((h,J), n_samples, iprint=iprint, rng=self.rng, **sampler_kw)
        self.sisj = np.concatenate(self.ising.corr[:2])
        self.p = self.ising.p
        self.allStates = self.ising.states.astype(np.int8)
        # determine p(k) as the number of votes in the plurality
        kVotes = list(map(lambda x:np.sort(np.bincount(x, minlength=3))[::-1],
                          self.allStates))
        self.coarseUix, self.coarseInvix = np.unique(kVotes, return_inverse=True, axis=0)
        self.coarseUix = np.unique(self.coarseInvix)
    
        if precompute:
            # cache triplet and quartet products
            if self.iprint: print("Starting correlations calculation...")
            self._triplets_and_quartets() 
            if self.iprint: print("Done.")

            if iprint: print("Computing dJ...")
            self.compute_dJ()
            if iprint: print("Done.")
        else:
            self.dJ = None
    
    def _observables_after_perturbation(self, si, sisj, i, a, eps):
        """Make spin i more like spin a.
        """
        
        n = self.n
        osi = si.copy()
        osisj = sisj.copy()
        
        # mimic average magnetization
        for k in range(self.kStates):
            si[i+k*n] = osi[i+k*n] - eps*(osi[i+k*n] - osi[a+k*n])

        for j in np.delete(list(range(n)),i):
            if i<j:
                ijix = unravel_index((i,j),n)
            else:
                ijix = unravel_index((j,i),n)

            if j==a:
                sisj[ijix] = osisj[ijix] - eps*(osisj[ijix] - 1)
            else:
                if j<a:
                    jaix = unravel_index((j,a),n)
                else:
                    jaix = unravel_index((a,j),n)
                sisj[ijix] = osisj[ijix] - eps*(osisj[ijix] - osisj[jaix])
    
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
        si = self.sisj[:n*3]
        sisj = self.sisj[3*n:]

        # observables after perturbations
        siNew = si.copy()
        sisjNew = sisj.copy()
        
        for i_, a_, eps_ in zip(i, a, eps):
            self._observables_after_perturbation(siNew, sisjNew, i_, a_, eps_)

        return np.concatenate((siNew, sisjNew))
  
    def compute_dJ(self, n_cpus=None):
        """Compute linear change to parameters for small perturbation.
        
        Parameters
        ----------
        n_cpus : int, None

        Returns
        -------
        dJ : ndarray
            (n_perturbation_parameters, n_maxent_parameters)
        """
        
        n_cpus = n_cpus or self.n_cpus

        def wrapper(params):
            """Make spin i more like spin a."""
            from numba import typed, types

            i, a = params

            # convert dicts to numba typed dicts for calc_A
            pairs = typed.Dict.empty(types.UniTuple(types.int64, 4), types.float64)
            for k,v in self.pairs.items():
                pairs[k] = v
            triplets = typed.Dict.empty(types.UniTuple(types.int64, 4), types.float64)
            for k,v in self.triplets.items():
                triplets[k] = v
            quartets = typed.Dict.empty(types.UniTuple(types.int64, 4), types.float64)
            for k,v in self.quartets.items():
                quartets[k] = v
            self.pairs, self.triplets, self.quartets = pairs, triplets, quartets

            return self.solve_linearized_perturbation(i, a)[0]
        
        # define fun for setting up args to pass into multiprocessing
        def args():
            for i in range(self.n):
                for a in np.delete(range(self.n), i):
                    yield (i, a)
        
        # run parallelized approach to solving perturbations
        if n_cpus is None or n_cpus > 1:
            if self.iprint: print("Multiprocessing for dJ...")
            with threadpool_limits(limits=1, user_api='blas'):
                with Pool(n_cpus) as pool:
                    dJ = np.vstack(( pool.map(wrapper, args()) ))
        else:
            dJ = np.zeros((self.n*(self.n-1), self.kStates*self.n+(self.n-1)*self.n//2))
            for counter, (i, a) in enumerate(args()):
                dJ[counter] = wrapper((i, a))

        self.dJ = dJ
        return dJ
  
    def _solve_linearized_perturbation_tester(self, iStar, aStar, full_output=False):
        """
        ***FOR DEBUGGING ONLY***

        Parameters
        ----------
        iStar : int
        aStar : int

        Returns
        -------
        ndarray
            Estimated linear change in maxent parameters.
        """
        
        n = self.n
        k = self.kStates
        p = self.p
        C = self.observables_after_perturbation(iStar, aStar)
        assert k==3, "Only handles k=3."

        from coniii.solvers import Enumerate
        from coniii.models import Potts3
        model = Potts3([np.zeros(k*n), np.zeros(n*(n-1)//2)])
        calc_observables = define_ternary_helper_functions()[1]
        solver = Enumerate(np.vstack((np.ones(n), -np.ones(n))),
                           model=model,
                           calc_observables=calc_observables)
       
        # hybr solver seems to work more consistently than default krylov
        hJ0 = solver.solve(constraints=self.sisj,
                           initial_guess=self.hJ,
                           full_output=True,
                           scipy_solver_kwargs={'method':'hybr', 'tol':1e-12})[0]

        # hybr solver seems to work more consistently than default krylov
        fullsoln = solver.solve(constraints=C,
                                initial_guess=self.hJ,
                                full_output=True,
                                scipy_solver_kwargs={'method':'hybr', 'tol':1e-12})
        soln = fullsoln[0]

        # remove translational offset for last set of fields
        hJ0[:n*k] -= np.tile(hJ0[n*(k-1):n*k], k)
        soln[:n*k] -= np.tile(soln[n*(k-1):n*k], k)
        if full_output:
            return (soln - self.hJ)/(self.eps), fullsoln
        return (soln - hJ0)/(self.eps)

    def _solve_linearized_perturbation(self, iStar, aStar,
                                       full_output=False,
                                       eps=None,
                                       check_stability=True):
        """Consider a perturbation to make spin i more like another spin a. 
        
        Parameters
        ----------
        iStar : int
        aStar : int
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
            (Aplus, Cplus)
        float (optional)
            Relative error to log10.
        """
        
        eps = eps or self.eps
        n = self.n
        kStates = self.kStates
        p = self.p
        si = self.sisj[:n*kStates]
        sisj = self.sisj[kStates*n:]

        # matrix that will be multiplied by the vector of correlation perturbations
        Cplus = self.observables_after_perturbation(iStar, aStar, eps=eps)
        Cminus = self.observables_after_perturbation(iStar, aStar, eps=-eps)
        errflag = 0
        
        if type(self.pairs) is dict:
            warn("Using slower version of calc_A.")
            Aplus = calc_A(n, kStates,
                           self.allStates, p, si, sisj,
                           self.pairs, self.triplets, self.quartets,
                           Cplus)
            Aminus = calc_A(n, kStates,
                           self.allStates, p, si, sisj,
                           self.pairs, self.triplets, self.quartets,
                           Cminus)
        else:
            Aplus = jit_calc_A(n, kStates,
                               self.allStates, p, si, sisj,
                               self.pairs, self.triplets, self.quartets,
                               Cplus)
            Aminus = jit_calc_A(n, kStates,
                                self.allStates, p, si, sisj,
                                self.pairs, self.triplets, self.quartets,
                                Cminus)
        Cplus -= self.sisj
        Cminus -= self.sisj
        # factor out linear dependence on eps
        dJ = np.linalg.lstsq(Aplus+Aminus, Cplus-Cminus, rcond=None)[0]/eps
        # put back in fields that we've fixed
        dJ = np.insert(dJ, (kStates-1)*n, np.zeros(n))

        if check_stability:
            # double epsilon and make sure solution does not change by a large amount
            dJtwiceEps, errflag = self._solve_linearized_perturbation(iStar, aStar,
                                                                      eps=eps/2,
                                                                      check_stability=False)
            # print if relative change is more than .1% for any entry excepting zeros
            # which are set by zeroed
            # fields
            zeroix = dJ==0
            relerr = np.log10(np.abs(dJ[~zeroix]-dJtwiceEps[~zeroix])) - np.log10(np.abs(dJ[~zeroix]))
            if (relerr>-3).any():
                if self.iprint:
                    print("Unstable solution. Recommend shrinking eps. Max err=%E"%(10**relerr.max()))
                errflag = 2
        
        if np.linalg.cond(Aplus)>1e15:
            warn("A is badly conditioned for pair (i, a)=(%d, %d)."%(iStar,aStar))
            # this takes precedence over relerr over threshold
            errflag = 1

        if full_output:
            if check_stability:
                return dJ, errflag, (Aplus, Cplus), relerr
            return dJ, errflag, (Aplus, Cplus)
        return dJ, errflag
#end Coupling3



class CanonicalMag3(Mag3):
    """Perturbation of fields."""
    def compute_dJ(self, n_cpus=None):
        """Compute linear change to parameters for small perturbation. These are trivial
        for the canonical case because each field is simply moved by a unit amount.
        
        Parameters
        ----------
        n_cpus : int, None

        Returns
        -------
        dJ : ndarray
            (n_perturbation_parameters, n_maxent_parameters)
        """
        
        n_cpus = n_cpus or self.n_cpus
        n = self.n
        
        # alter each field in sequence
        dJ = np.hstack((np.eye(3*n), np.zeros((3*n, n*(n-1)//2))))
        self.dJ = dJ

        return dJ
#end CanonicalMag3



class CanonicalCoupling3(Coupling3):
    """Perturbation of fields."""
    def compute_dJ(self, n_cpus=None):
        """Compute linear change to parameters for small perturbation. These are trivial
        for the canonical case because each field is simply moved by a unit amount.
        
        Parameters
        ----------
        n_cpus : int, None

        Returns
        -------
        dJ : ndarray
            (n_perturbation_parameters, n_maxent_parameters)
        """
        
        n_cpus = n_cpus or self.n_cpus
        n = self.n
        
        dJ = np.hstack((np.zeros((n*(n-1)//2, n*3)), np.eye(n*(n-1)//2)))
        self.dJ = dJ

        return dJ
#end CanonicalCoupling3



# ============= #
# JIT functions #
# ============= #
@njit("float64(float64[:],int8[:])")
def fast_sum(J, s):
    """Helper function for calculating energy in calc_e(). Iterates couplings J."""
    e = 0
    k = 0
    for i in range(len(s)-1):
        for j in range(i+1,len(s)):
            e += J[k]*s[i]*s[j]
            k += 1
    return e

@njit("float64(float64[:],int8[:])")
def fast_sum_ternary(J, s):
    """Helper function for calculating energy in calc_e(). Iterates couplings J."""
    assert len(J)==(len(s)*(len(s)-1)//2)

    e = 0
    k = 0
    for i in range(len(s)-1):
        for j in range(i+1,len(s)):
            if s[i]==s[j]:
                e += J[k]
            k += 1
    return e

@njit("float64[:](int64,int64,int8[:,:],float64[:])")
def calc_all_energies(n, k, states, params):
    """Calculate all the energies for the states given. Can be used for Potts.
    
    Parameters
    ----------
    n : int
        Number of spins.
    k : int
        Ising or Potts3 model.
    states : ndarray
        Number of distinct states.
    params : ndarray
        (h,J) vector

    Returns
    -------
    E : ndarray
        Energies of all given states.
    """
    
    e = np.zeros(len(states))
    s_ = np.zeros((1,n), dtype=np.int8)
    if k==2:
        for i in range(len(states)):
            s = states[i]
            e[i] -= fast_sum(params[n:], s)
            e[i] -= np.sum(s*params[:n])
    elif k==3:
        for i in range(len(states)):
            s = states[i]
            for ix in range(n):
                # fields
                e[i] -= params[ix+s[ix]*n]
            e[i] -= fast_sum_ternary(params[n*k:], s)
    else: raise NotImplementedError
    return e

def jit_spin_replace_transition_matrix(n, i, j, eps):
    rows = []
    cols = []
    vals = []
    for ix in range(2**n):
        s = bin(ix)[2:].zfill(n)
        if s[i]!=s[j]:
            rows.append(ix)
            cols.append(ix)
            vals.append(1-eps)
            
            if s[i]=='0':
                s = s[:i]+'1'+s[i+1:]
            else:
                s = s[:i]+'0'+s[i+1:]
            rows.append(int(s,2))
            cols.append(ix)
            vals.append(eps)
        else:
            rows.append(ix)
            cols.append(ix)
            vals.append(1.)
    return rows, cols, vals

@njit(cache=True)
def fast_sum(J,s):
    """Helper function for calculating energy in calc_e(). Iterates couplings J."""
    e = np.zeros((s.shape[0]))
    for n in range(s.shape[0]):
        k = 0
        for i in range(s.shape[1]-1):
            for j in range(i+1,s.shape[1]):
                e[n] += J[k]*s[n,i]*s[n,j]
                k += 1
    return e

@njit("float64[:](int8[:,:],float64[:])")
def calc_e(s, params):
    """Energy of Ising model.

    Parameters
    ----------
    s : 2D ndarray
        state either {0,1} or {+/-1}
    params : ndarray
        (h,J) vector

    Returns
    -------
    E : ndarray
        Energies of all given states.
    """
    
    e = -fast_sum(params[s.shape[1]:],s)
    e -= np.sum(s*params[:s.shape[1]],1)
    return e

@njit
def jit_pair_combination(n):
    for i in range(n-1):
        for j in range(i+1,n):
            yield i,j

@njit(cache=True)
def jit_triplets(n, kStates, allStates, p):
    """Calculate pairwise, triplet correlations.
    """
    pairs = dict()
    triplets = dict()

    # <d_{i,gammai} * d_{j,gammaj}> where i<j
    for i,j in jit_pair_combination(n):
        for gammai in range(kStates):
            for gammaj in range(kStates):
                pairs[(gammai,i,gammaj,j)] = sum_single_cols(p, allStates, i, gammai, j, gammaj)

    # triplets that matter are when one spin is in a particular state and the
    # remaining two agree with each other
    for gamma in range(kStates):
        for i in range(n):
            for j,k in jit_pair_combination(n):
                triplets[(gamma,i,j,k)] = sum_col_pair(p, allStates, i, gamma, j, k)

    return pairs, triplets

@njit(cache=True)
def jit_triplets_and_quartets(n, kStates, allStates, p):
    """Calculate pairwise, triplet, and quartet correlations.
    """
    pairs = dict()
    triplets = dict()
    quartets = dict()

    # <d_{i,gammai} * d_{j,gammaj}> where i<j
    for i,j in jit_pair_combination(n):
        for gammai in range(kStates):
            for gammaj in range(kStates):
                pairs[(gammai,i,gammaj,j)] = sum_single_cols(p, allStates, i, gammai, j, gammaj)

    # triplets that matter are when one spin is in a particular state and the
    # remaining two agree with each other
    for gamma in range(kStates):
        for i in range(n):
            for j,k in jit_pair_combination(n):
                triplets[(gamma,i,j,k)] = sum_col_pair(p, allStates, i, gamma, j, k)

    # quartets that matter are when the first pair are the same and the second pair
    # are the same
    for i,j in jit_pair_combination(n):
        for k,l in jit_pair_combination(n):
            quartets[(i,j,k,l)] = sum_pair_pair(p, allStates, i, j, k, l)

    return pairs, triplets, quartets

@njit(cache=True)
def jit_calc_A(n, kStates, allStates, p, si, sisj, pairs, triplets, quartets, C):
    """Calculate matrix A in the linearized problem for a specific given perturbation
    to the means and pairwise correlations captured in vector C.

    The linear problem that will be solved for dJ is
        A * dJ = C

    Parameters
    ----------
    n : int
    kStates : int
    allStates : ndarray
    p : ndarray
    si : ndarray
    sisj : ndarray
    pairs : numba.dict
    triplets : numba.dict
    quartets : numba.dict
    C : ndarray

    Returns
    -------
    ndarray
        The matrix A.
    """

    A = np.zeros((kStates*n+n*(n-1)//2, (kStates-1)*n+n*(n-1)//2))

    # mean constraints corresponding to odd order correlations 
    # remember that A does not include changes in last set of fields (corresponding to the
    # last Potts state)
    # i is the index of the perturbed spin
    for i in prange(kStates*n):
        for j in prange((kStates-1)*n):
            if i==j:
                # p(s_i=gamma) - p(s_i=gamma) * p(s_i=zeta)
                A[i,j] = si[i] - C[i]*si[i]
            elif (i%n)==(j%n):  # if they're in different states but the same spin
                A[i,j] = -C[i]*si[j]
            else:  # if they're different spins in different states
                if (i%n)<(j%n):  # just because there is a fixed ordering to the indexing of pairs
                    A[i,j] = pairs[(i//n,i%n,j//n,j%n)] - C[i] * si[j]
                else:
                    A[i,j] = pairs[(j//n,j%n,i//n,i%n)] - C[i] * si[j]

        for klcount,(k,l) in enumerate(jit_pair_combination(n)):
            A[i,(kStates-1)*n+klcount] = triplets[(i//n,i%n,k,l)] - C[i] * sisj[klcount]
    
    # pair constraints
    for ijcount,(i,j) in enumerate(jit_pair_combination(n)):
        for k in prange((kStates-1)*n):
            A[kStates*n+ijcount,k] = triplets[(k//n,k%n,i,j)] - C[kStates*n+ijcount] * si[k]

        for klcount,(k,l) in enumerate(jit_pair_combination(n)):
            A[kStates*n+ijcount,(kStates-1)*n+klcount] = (quartets[(i,j,k,l)] -
                                                          C[kStates*n+ijcount] *
                                                          sisj[klcount])
    return A

def calc_A(n, kStates, allStates, p, si, sisj, pairs, triplets, quartets, C):
    """Calculate matrix A in the linearized problem for a specific given perturbation
    to the means and pairwise correlations captured in vector C.

    The linear problem that will be solved for dJ is
        A * dJ = C

    Parameters
    ----------
    n : int
    kStates : int
    allStates : ndarray
    p : ndarray
    si : ndarray
    sisj : ndarray
    pairs : dict
    triplets : dict
    quartets : dict
    C : ndarray

    Returns
    -------
    ndarray
        The matrix A.
    """

    A = np.zeros((kStates*n+n*(n-1)//2, (kStates-1)*n+n*(n-1)//2))

    # mean constraints corresponding to odd order correlations 
    # remember that A does not include changes in last set of fields (corresponding to the
    # last Potts state)
    # i is the index of the perturbed spin
    for i in prange(kStates*n):
        for j in prange((kStates-1)*n):
            if i==j:
                # p(s_i=gamma) - p(s_i=gamma) * p(s_i=zeta)
                A[i,j] = si[i] - C[i]*si[i]
            elif (i%n)==(j%n):  # if they're in different states but the same spin
                A[i,j] = -C[i]*si[j]
            else:  # if they're different spins in different states
                if (i%n)<(j%n):  # just because there is a fixed ordering to the indexing of pairs
                    A[i,j] = pairs[(i//n,i%n,j//n,j%n)] - C[i] * si[j]
                else:
                    A[i,j] = pairs[(j//n,j%n,i//n,i%n)] - C[i] * si[j]

        for klcount,(k,l) in enumerate(jit_pair_combination(n)):
            A[i,(kStates-1)*n+klcount] = triplets[(i//n,i%n,k,l)] - C[i] * sisj[klcount]
    
    # pair constraints
    for ijcount,(i,j) in enumerate(jit_pair_combination(n)):
        for k in prange((kStates-1)*n):
            A[kStates*n+ijcount,k] = triplets[(k//n,k%n,i,j)] - C[kStates*n+ijcount] * si[k]

        for klcount,(k,l) in enumerate(jit_pair_combination(n)):
            A[kStates*n+ijcount,(kStates-1)*n+klcount] = (quartets[(i,j,k,l)] -
                                                          C[kStates*n+ijcount] *
                                                          sisj[klcount])
    return A

@njit("float64(float64[:],int8[:,:],int64,int64,int64,int64)", parallel=True)
def sum_single_cols(p, allStates, col1, col1val, col2, col2val):
    """Two columns that take different values.
    """
    total = 0.

    for i in prange(p.size):
        if allStates[i,col1]==col1val and allStates[i,col2]==col2val:
            total += p[i]

    return total

@njit("float64(float64[:],int8[:,:],int64,int64,int64,int64)", parallel=True)
def sum_col_pair(p, allStates, col1, col1val, col2, col3):
    """One col that takes some value and two cols in agreement.
    """
    total = 0.

    for i in prange(p.size):
        if allStates[i,col1]==col1val and allStates[i,col2]==allStates[i,col3]:
            total += p[i]

    return total

@njit("float64(float64[:],int8[:,:],int64,int64,int64,int64)", parallel=True)
def sum_pair_pair(p, allStates, col1, col2, col3, col4):
    """Two separate cols in agreement with one another.
    """
    total = 0.

    for i in prange(p.size):
        if allStates[i,col1]==allStates[i,col2] and allStates[i,col3]==allStates[i,col4]:
            total += p[i]

    return total
