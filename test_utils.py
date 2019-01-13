# ============================================================================================ #
# Quick access to useful modules from SCOTUS project.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ============================================================================================ # 
from .utils import *


def test_tweak_constraints(n=3):
    X=np.random.choice([-1,1],(100,n))
    sisj=pair_corr(X, concat=True)
    for i in range(n):
        for j in delete(arange(n),i):
            Xdup=X.copy()
            Xdup[:,i]=X[:,j]
            assert np.isclose( pair_corr(Xdup, concat=True),
                               tweak_constraints(sisj, i, j, 1, n) ).all()
