# Module for timing.
# Author: Eddie Lee, edl56@cornell.edu
from .fim import *
import timeit 
n = 6
hJ = np.concatenate((np.zeros(n), np.random.normal(scale=.3, size=n*(n-1)//2)))


def example():
    isingdkl = IsingFisherCurvatureMethod2(n, h=hJ[:n], J=hJ[n:])
    isingdkl.maj_curvature(n_cpus=1)
