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
