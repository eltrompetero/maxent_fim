# ============================================================================================ #
# Test module for coarse_grain.py
# Author : Eddie Lee, edlee@santafe.edu
# ============================================================================================ # 
from .coarse_grain import *



def test_enumerate_unique_splits():
    for i in range(3, 22, 3):
        splits = enumerate_unique_splits(i)
        assert (splits.sum(1)==i).all()
        print(f"Test passed {i}: splits add up correctly.")

        assert (np.diff(splits[:,0])<=0).all()
        print(f"Test passed {i}: largest group monotonically decreases in specified order.")

        assert (np.diff(splits, axis=1)<=0).all()
        print(f"Test passed {i}: Preceding groups are always larger than the next.")

    for i in range(3, 22, 3):
        splits = enumerate_unique_splits(i, 2)
        assert (splits.sum(1)==i).all()
        print(f"Test passed {i}: splits add up correctly.")

        assert (np.diff(splits[:,0])<=0).all()
        print(f"Test passed {i}: largest group monotonically decreases in specified order.")

        assert (np.diff(splits, axis=1)<=0).all()
        print(f"Test passed {i}: Preceding groups are always larger than the next.")
