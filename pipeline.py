# =============================================================================================== #
# Pipeline for pivotal voter analysis.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
import numpy as np
import os
import pickle
import dill
from .utils import *
from .fim import *


def entropy_estimates(system, method,
                      prefix='cache',
                      n_boot=10_000,
                      Squad_kwargs={}):
    """
    Parameters
    ----------
    system : str
    method : str
    prefix : str, 'cache'
        Constructed path will be of form '%s/%s/%s'%(prefix,system,method)
    n_boot : int, 10_000

    Returns
    -------
    ndarray
        miCaptured
    ndarray
        Entropy of data.
    ndarray
        Entropy of indpt model.
    ndarray
        Entropy of pairwise fit.
    ndarray
        Ma entropy.
    ndarray
        Quadratic fit.
    """
    
    from entropy.estimators import S_ma

    path = 'cache/%s/%s'%(system,method)

    fname = '%s/data.p'%path
    if os.path.isfile(fname):
        data = pickle.load(open(fname,'rb'))['data']

        miCaptured = np.zeros(len(data))
        Sdata = np.zeros(len(data))
        Sind = np.zeros(len(data))
        Spair = np.zeros(len(data))
        Sma = np.zeros(len(data))
        quadfit = []

        for i,k in enumerate(list(data.keys())):
            if len(data[k])>2 and (not data[k][-1] is None) and data[k][-1]['success']:
                X, hJ = data[k][1:3]
                assert (X.mean(0)==0).all()
                miCaptured[i], Stuple, quadfit_ = multi_info(X, hJ, n_boot=n_boot, disp=False, **Squad_kwargs)
                quadfit.append(quadfit_)
                Sdata[i], Sind[i], Spair[i] = Stuple
                Sma[i] = S_ma(X)

    return miCaptured, Sdata, Sind, Spair, Sma, quadfit

def check_corr(system, method, orders=range(2,9,2), prefix='cache'):
    """
    Parameters
    ----------
    system : str
    method : str
    prefix : str, 'cache'
        Constructed path will be of form '%s/%s/%s'%(prefix,system,method)
    n_boot : int, 10_000

    Returns
    -------
    ndarray
        Errors on each kth order correlation coefficient.
    ndarray
        Correlation coefficient of model estimate of correlation by data.
    """
    
    import importlib

    path = 'cache/%s/%s'%(system,method)

    fname = '%s/data.p'%path
    if os.path.isfile(fname):
        data = pickle.load(open(fname,'rb'))['data']
        errs = np.zeros((len(data),len(orders)))
        corr = np.zeros((len(data),len(orders)))

        for i,k in enumerate(list(data.keys())):
            if len(data[k])>2 and (not data[k][-1] is None) and data[k][-1]['success']:
                X, hJ = data[k][1:3]
                assert (X.mean(0)==0).all()
                ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%X.shape[1])
                p = ising.p(hJ)
                
                errs[i], corr[i] = check_correlations(X, p, orders)
    return errs, corr

def solve_inverse_on_data(data, n_cpus=4, potts=False):
    """Automate solution of inverse problem on data dictionary. Only run on tuples in dict
    that only have two entries (the others presumably have already been solved and the
    solutions saved).

    Turns out that Pseudo can help get close enough to the solution when Enumerate is
    having a hard time calculating the gradient accurately. This happens (surprisingly)
    in weird parts of parameter space where the couplings are not necessarily much larger
    than unity.

    Parameters
    ----------
    data : dict
        As saved into the data dictionary when reading out voting records. Each element is
        a two-element list containing the voter names, voting record.
    n_cpus : int, 4

    Returns
    -------
    None
    """
    
    from coniii.utils import pair_corr, define_ising_helper_functions, define_pseudo_ising_helper_functions
    from data_sets.neuron.data import potts_pair_corr
    from coniii.solvers import Enumerate, Pseudo
    import importlib
    from multiprocess import Pool, cpu_count
    
    _, calc_observables, _ = define_ising_helper_functions()

    def single_solution_wrapper(item):
        name = item[0]
        X = item[1][1]
        n = X.shape[1]
        print("Solving %s."%name)
        if potts:
            assert np.array_equal(np.unique(X), [0,1,2])
            sisj = potts_pair_corr(X, k=3, concat=True)
            ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_potts'%n)
        else:
            sisj = pair_corr(X, concat=True)
            ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        enumSolver = Enumerate(n, calc_observables_multipliers=ising.calc_observables)

        hJ, soln = enumSolver.solve(sisj,
                                    max_param_value=50*n/9,
                                    full_output=True,
                                    scipy_solver_kwargs={'method':'hybr'})
        if not potts and np.linalg.norm(soln['fun'])>1e-3:
            # try Pseudo (non-ergodic?)
            print("Entering pseudo %s."%name)
            get_multipliers_r, calc_observables_r = define_pseudo_ising_helper_functions(n)
            pseudoSolver = Pseudo(n,
                                  calc_observables=calc_observables,
                                  calc_observables_r=calc_observables_r,
                                  get_multipliers_r=get_multipliers_r)
            hJ = pseudoSolver.solve(X, np.zeros(n+n*(n-1)//2))

            # try again
            try:
                hJ, soln = enumSolver.solve(sisj,
                                            initial_guess=hJ,
                                            max_param_value=50*n/9,
                                            method={'method':'hybr'},
                                            full_output=True)
            # this occurs when Jacobian inverse returns zero vector
            except ValueError:
                soln = None
        elif potts and np.linalg.norm(soln['fun'])>1e-3:
            hJ, soln = enumSolver.solve(sisj,
                                        max_param_value=50*n/9,
                                        full_output=True,
                                        scipy_solver_kwargs={'method':'krylov'})

        print("Done %s."%name)
        return hJ, soln
    
    if n_cpus>1:
        pool = Pool(4)
        hJ, soln = list( zip(*pool.map(single_solution_wrapper, 
                               [i for i in data.items() if len(i[1])==2] )))
        pool.close()
    else:
        hJ = []
        soln = []
        for item in [i for i in data.items() if len(i[1])==2]:
            hJ_, soln_ = single_solution_wrapper(item)
            hJ.append(hJ_)
            soln.append(soln_)
    
    # update data dict
    keys = [i[0] for i in data.items() if len(i[1])==2]
    for i,k in enumerate(keys):
        n = len(data[k][0])
        if potts:
            hJ[i][:n*3] -= np.tile(hJ[i][:n],3)
        data[k].append(hJ[i])
        data[k].append(soln[i])
    assert all([len(i)==4 for i in data.values()])

def calculate_fisher_on_pk(data, system, method,
                           computed_results=None,
                           high_prec_dps=30,
                           save=True,
                           save_every_loop=True,
                           fi_method=2,
                           high_prec=False):
    """
    Parameters
    ----------
    data : dict
    system : str
    method : str
    computed_results: dict, None
        If given, results will be appended onto this.
    high_prec_dps : int, 30
    save : bool, True
        Save to pickle.
    save_every_loop : bool, True
        If False, only save at very end after loops.
    fi_method : int, 2
    allow_high_prec : bool, True
        If True, allow high precision calculation to run.

    Returns
    -------
    dict
    """
    
    import importlib

    fname = 'cache/Method%s/%s/%s/fisherResultMaj.p'%(str(fi_method),system,method)
    if not os.path.isdir('cache/Method%s/%s/%s'%(str(fi_method),system,method)):
        os.makedirs('cache/Method%s/%s/%s'%(str(fi_method),system,method))
    
    if computed_results is None:
        fisherResultMaj = {}
    else:
        fisherResultMaj = computed_results

    for k in [kp for kp in data.keys() if not kp in fisherResultMaj.keys()]:
        if (data[k][-1] is None or
            (np.linalg.norm(data[k][-1]['fun'])<1e-6)):
            print("Starting %s..."%k)
            n = len(data[k][0])
            hJ = data[k][2]
            
            try:
                if str(fi_method)=='1':
                    isingdkl = IsingFisherCurvatureMethod1(n, h=hJ[:n], J=hJ[n:], eps=1e-6, high_prec=high_prec)
                elif fi_method=='1a':
                    isingdkl = IsingFisherCurvatureMethod1a(n, h=hJ[:n], J=hJ[n:], eps=1e-6)
                elif str(fi_method)=='2':
                    isingdkl = IsingFisherCurvatureMethod2(n, h=hJ[:n], J=hJ[n:], eps=1e-6)
                elif str(fi_method)=='2b':
                    isingdkl = IsingSpinReplacementFIM(n, h=hJ[:n], J=hJ[n:], eps=1e-6)
                elif str(fi_method)=='3':
                    isingdkl = IsingFisherCurvatureMethod3(n, h=hJ[:n], J=hJ[n:], eps=1e-6)
                elif str(fi_method)=='4':
                    isingdkl = IsingFisherCurvatureMethod4(n, 3, h=hJ[:n*3], J=hJ[3*n:], eps=1e-6)
                elif fi_method=='4a':
                    isingdkl = IsingFisherCurvatureMethod4a(n, 3, h=hJ[:n*3], J=hJ[3*n:], eps=1e-6)
                else:
                    raise Exception("Invalid method.")
                if fi_method=='2b':
                    hess, errflag, err = isingdkl.maj_curvature(full_output=True, epsdJ=isingdkl.eps)
                else:
                    epsdJ = min(1/np.abs(isingdkl.dJ).max()/10, 1e-4)
                    hess, errflag, err = isingdkl.maj_curvature(full_output=True, epsdJ=epsdJ)

                eigval, eigvec = isingdkl.hess_eig(hess)
                
                fisherResultMaj[k] = [isingdkl, (hess, errflag, err), eigval, eigvec]
            
                if save and save_every_loop:
                    print("Saving into %s"%fname)
                    with open(fname, 'wb') as f:
                        dill.dump({'fisherResultMaj':fisherResultMaj}, f, -1)
            except AssertionError as e:
                print("AssertionError for key %s"%k)
                print(e)

    if save and not save_every_loop:
        print("Saving into %s"%fname)
        with open(fname, 'wb') as f:
            dill.dump({'fisherResultMaj':fisherResultMaj}, f, -1)
    return fisherResultMaj

def extract_voter_subspace(fisherResult,
                           return_n_voters=3,
                           remove_n_modes=0):
    """
    Parameters
    ----------
    fisherResult : dict
    return_n_voters : int, False
        If an int is given, number of voter subspace eigenvalues to return.
    remove_n_modes : int, 0
        If True, subtract off principal modes from Hessian.

    Returns
    -------
    ndarray
        Principal bloc
    ndarray
        Voter subspace eigenvalues (sorted).
    ndarray
        sort index
    """
    
    K = len(fisherResult)
    pivotalEigval = np.zeros(K)-1
    voterEigval = np.zeros((K,return_n_voters))
    voterEigvalSortix = np.zeros((K,return_n_voters), dtype=int)

    for i,k in enumerate(fisherResult.keys()):
        out = _extract_voter_subspace(fisherResult[k], return_n_voters, remove_n_modes) 
        pivotalEigval[i] = out[0] 
        voterEigval[i] = out[1]
        voterEigvalSortix[i] = out[2]

        assert 0<=(voterEigval[i,0]/pivotalEigval[i])<=1, "Hessian calculation error. Condition violated."
    return pivotalEigval, voterEigval, voterEigvalSortix

def _extract_voter_subspace(fisherResultValue,
                            return_n_voters=3,
                            remove_n_modes=0):
    """
    Parameters
    ----------
    fisherResultValue : list
    return_n_voters : int, False
        If an int is given, number of voter subspace eigenvalues to return.
    remove_n_modes : int, 0
        If True, subtract off n principal modes from Hessian.

    Returns
    -------
    ndarray
        Principal bloc
    ndarray
        Voter subspace eigenvalues (sorted).
    ndarray
        sort index
    """
    
    voterEigval = np.zeros(return_n_voters)
    voterEigvalSortix = np.zeros(return_n_voters, dtype=int)
    n = fisherResultValue[0].n
    
    # read out results stored in dict
    isingdkl, (hess, errflag, err), eigval, eigvec = fisherResultValue
    if remove_n_modes>0:
        for i in range(remove_n_modes):
            hess = remove_principal_mode(hess)
        eigval, eigvec = np.linalg.eig(hess)
        sortix = np.argsort(eigval)[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
    
    # only consider hessians that are well-estimated
    pivotalEigval = eigval[0]

    # when limited to the subspace of a single voter at a given time (how do we 
    # optimally tweak a single voter to change the system?)
    veigval = []
    veigvec = []
    
    
    if type(fisherResultValue[0]) is IsingFisherCurvatureMethod1:
        for j in range(n):
            veigval.append(hess[j,j])
            veigvec.append(np.ones(1))
    elif type(fisherResultValue[0]) is IsingFisherCurvatureMethod4a:
        for j in range(n):
            subspaceHess = hess[[j,j+n,j+2*n],:][:,[j,j+n,j+2*n]]
            u, v = np.linalg.eig(subspaceHess)
            sortix = np.argsort(u)[::-1]
            u = u[sortix]
            v = v[:,sortix]

            veigval.append(u)
            veigvec.append(v)
    elif not type(fisherResultValue[0]) is IsingFisherCurvatureMethod4a:
        # iterate through subspace for each voter (assuming each voter is connected n-1 others
        for j in range(n):
            subspaceHess = hess[j*(n-1):(j+1)*(n-1), j*(n-1):(j+1)*(n-1)]
            u, v = np.linalg.eig(subspaceHess)
            sortix = np.argsort(u)[::-1]
            u = u[sortix]
            v = v[:,sortix]

            veigval.append(u)
            veigvec.append(v)
    voterEigval_ = np.vstack(veigval)

    # sort voters by largest voter eigenvalue
    voterEigvalSortix = np.argsort(voterEigval_[:,0])[::-1][:return_n_voters]
    voterEigval = voterEigval_[:,0][voterEigvalSortix]

    assert 0<=(voterEigval[0]/pivotalEigval)<=1, "Hessian calculation error. Condition violated."
    return pivotalEigval, voterEigval, voterEigvalSortix

def degree_collective(fisherResult, **kwargs):
    """
    Parameters
    ----------
    fisherResult : dict
    method : str, 'eig'
        'val': use individual subspace eigenvalues
        'vec': use weight in entries of eigenvector
    remove_first_mode : bool, False
        If True, subtract off principal mode from Hessian.

    Returns
    -------
    ndarray
    """
    
    K = len(fisherResult)
    degree = np.zeros(K)-1
    
    for i,k in enumerate(fisherResult.keys()):
        if type(fisherResult[k][0]) is IsingFisherCurvatureMethod1:
            degree[i] = _degree_collective1(fisherResult[k], **kwargs) 
        elif type(fisherResult[k][0]) is IsingFisherCurvatureMethod2:
            degree[i] = _degree_collective2(fisherResult[k], **kwargs) 
        elif type(fisherResult[k][0]) is IsingFisherCurvatureMethod4:
            degree[i] = _degree_collective2(fisherResult[k], **kwargs) 
        elif type(fisherResult[k][0]) is IsingFisherCurvatureMethod4a:
            degree[i] = _degree_collective4a(fisherResult[k], **kwargs) 
        else:
            raise Exception("Invalid type for key %s."%k)
    return degree

def _degree_collective1(fisherResultValue,
                        remove_n_modes=0,
                        voter_eig_rank=0,
                        method='val'):
    """
    Parameters
    ----------
    fisherResultValue : list
    remove_first_mode : bool, False
        If True, subtract off principal mode from rows of Hessian.
    voter_eig_rank : int, 0
        Rank of eigenvalue and eigenvector to return from voter subspaces.
    method : str, 'val'

    Returns
    -------
    float
        If 'val' option, then the entropy of the sum of the columns is returned.
        If 'vec' option, then the fractional weights per column are returned.
    """
    
    n = fisherResultValue[0].n
    isingdkl, (hess, errflag, err), eigval, eigvec = fisherResultValue
    if remove_n_modes>0:
        for i in range(remove_n_modes):
            hess = remove_principal_mode(hess)
        eigval, eigvec = np.linalg.eig(hess)
        sortix = np.argsort(eigval)[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
    
    # only consider hessians that are well-estimated
    #if err is None or np.linalg.norm(err)<(.05*np.linalg.norm(hess)):
    if method=='vec':
        p = eigvec[:,0]**2
        p /= p.sum()

    elif method=='val':
        # when limited to the subspace of a single voter at a given time (how do we 
        # optimally tweak a single voter to change the system?)
        veigval = []
        veigvec = []
        
        # iterate through subspace for each voter (assuming each voter is connected n-1 others
        for j in range(n):
            veigval.append(hess[j,j])
            veigvec.append(np.ones(1))
        veigval = np.vstack(veigval)[:,voter_eig_rank]
        
        # entropy
        p = veigval / veigval.sum()

    else:
        raise Exception("Invalid choice for method.")
    degree = -np.log2(p).dot(p) / np.log2(p.size)
    return degree

def _degree_collective2(fisherResultValue,
                        remove_n_modes=0,
                        voter_eig_rank=0,
                        method='val'):
    """
    Parameters
    ----------
    fisherResultValue : list
    remove_first_mode : bool, False
        If True, subtract off principal mode from rows of Hessian.
    voter_eig_rank : int, 0
        Rank of eigenvalue and eigenvector to return from voter subspaces.
    method : str, 'val'

    Returns
    -------
    float
        If 'val' option, then the entropy of the sum of the columns is returned.
        If 'vec' option, then the fractional weights per column are returned.
    """
    
    n = fisherResultValue[0].n
    isingdkl, (hess, errflag, err), eigval, eigvec = fisherResultValue
    if remove_n_modes>0:
        for i in range(remove_n_modes):
            hess = remove_principal_mode(hess)
        eigval, eigvec = np.linalg.eig(hess)
        sortix = np.argsort(eigval)[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
    
    # only consider hessians that are well-estimated
    #if err is None or np.linalg.norm(err)<(.05*np.linalg.norm(hess)):
    if method=='vec':
        v = np.insert(eigvec[:,0], range(0,n*n,n), 0).reshape(n,n)
        #p = (v**2).sum(1)
        #p = ((v**2).sum(1)+(v**2).sum(0))/2
        #p /= p.sum()
        degree = (v**2+v.T**2-2*np.abs(v*v.T)).sum()/2

    elif method=='val':
        # when limited to the subspace of a single voter at a given time (how do we 
        # optimally tweak a single voter to change the system?)
        veigval = []
        veigvec = []
        
        # iterate through subspace for each voter (assuming each voter is connected n-1 others
        for j in range(n):
            subspaceHess = hess[j*(n-1):(j+1)*(n-1), j*(n-1):(j+1)*(n-1)]
            u, v = np.linalg.eig(subspaceHess)
            sortix = np.argsort(u)[::-1]
            u = u[sortix]
            v = v[:,sortix]

            veigval.append(u)
            veigvec.append(v)
        veigval = np.vstack(veigval)[:,voter_eig_rank]
        
        # entropy
        p = veigval / veigval.sum()

    else:
        raise Exception("Invalid choice for method.")
    #degree = -np.log2(p).dot(p) / np.log2(p.size)
    return degree

def _degree_collective4a(fisherResultValue,
                         remove_n_modes=0,
                         voter_eig_rank=0,
                         method='val'):
    """
    Parameters
    ----------
    fisherResultValue : list
    remove_first_mode : bool, False
        If True, subtract off principal mode from rows of Hessian.
    voter_eig_rank : int, 0
        Rank of eigenvalue and eigenvector to return from voter subspaces.
    method : str, 'val'

    Returns
    -------
    float
        If 'val' option, then the entropy of the sum of the columns is returned.
        If 'vec' option, then the fractional weights per column are returned.
    """
    
    n = fisherResultValue[0].n
    isingdkl, (hess, errflag, err), eigval, eigvec = fisherResultValue
    if remove_n_modes>0:
        for i in range(remove_n_modes):
            hess = remove_principal_mode(hess)
        eigval, eigvec = np.linalg.eig(hess)
        sortix = np.argsort(eigval)[::-1]
        eigval = eigval[sortix]
        eigvec = eigvec[:,sortix]
    
    # only consider hessians that are well-estimated
    #if err is None or np.linalg.norm(err)<(.05*np.linalg.norm(hess)):
    if method=='vec':
        v = eigvec[:,voter_eig_rank].reshape(3,n)
        p = (v**2).sum(0)
        p /= p.sum()

    elif method=='val':
        # when limited to the subspace of a single voter at a given time (how do we 
        # optimally tweak a single voter to change the system?)
        veigval = []
        veigvec = []
        
        # iterate through subspace for each voter (assuming each voter is connected n-1 others
        for j in range(n):
            subspaceHess = hess[[j,j+n,j+2*n],:][:,[j,j+n,j+2*n]]
            u, v = np.linalg.eig(subspaceHess)
            sortix = np.argsort(u)[::-1]
            u = u[sortix]
            v = v[:,sortix]

            veigval.append(u)
            veigvec.append(v)
        veigval = np.vstack(veigval)[:,voter_eig_rank]
        
        # entropy
        p = veigval / veigval.sum()

    else:
        raise Exception("Invalid choice for method.")
    degree = -np.log2(p).dot(p) / np.log2(p.size)
    return degree
