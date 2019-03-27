# =============================================================================================== #
# Pipeline for pivotal voter analysis.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
import numpy as np
import os
import pickle
import dill
from .utils import *


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
            if data[k][-1]['success']:
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
            if data[k][-1]['success']:
                X, hJ = data[k][1:3]
                assert (X.mean(0)==0).all()
                ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%X.shape[1])
                p = ising.p(hJ)
                
                errs[i], corr[i] = check_correlations(X, p, orders)
    return errs, corr

def solve_inverse_on_data(data, n_cpus=4):
    """Automate solution of inverse problem on data dictionary.

    Parameters
    ----------
    data : dict
    n_cpus : int, 4

    Returns
    -------
    None
    """
    
    from coniii.utils import pair_corr, define_pseudo_ising_helpers
    import importlib
    from multiprocess import Pool, cpu_count

    def single_solution_wrapper(item):
        name = item[0]
        X = item[1][1]
        n = X.shape[1]
        sisj = pair_corr(X, concat=True)

        print("Solving %s."%name)
        ising = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        enumSolver = Enumerate(n, calc_observables_multipliers=ising.calc_observables)

        hJ, soln = enumSolver.solve(sisj,
                                    max_param_value=50*n/9,
                                    full_output=True,
                                    scipy_solver_kwargs={'method':'hybr'})
        if np.linalg.norm(soln['fun'])>1e-3:
            # try Pseudo (non-ergodic?)
            print("Entering pseudo %s."%name)
            get_multipliers_r, calc_observables_r = define_pseudo_ising_helpers(n)
            pseudoSolver = Pseudo(n,
                                  calc_observables=calc_observables,
                                  calc_observables_r=calc_observables_r,
                                  get_multipliers_r=get_multipliers_r)
            hJ = pseudoSolver.solve(X, np.zeros(n+n*(n-1)//2), general_case=True)

            # try again
            hJ, soln = enumSolver.solve(sisj,
                                        initial_guess=hJ,
                                        max_param_value=50*n/9,
                                        full_output=True)
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
        data[k].append(hJ[i])
        data[k].append(soln[i])
    assert all([len(i)==4 for i in data.values()])
