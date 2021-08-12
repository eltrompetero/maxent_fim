# Fisher information of maxent models
Code to accompany preprint "Discovering sparse control strategies in C. elegans" 
(<http://arxiv.org/abs/2108.00837>) by Edward D. Lee, Xiaowen Chen and Bryan C. Daniels.

Computation for even modestly sized systems is expensive and best done on a large
computing cluster. Cached results from computations are plotted in the accompanying
Jupyter notebook.

Given the number of calculations done, the code includes a wrapper defined in
`organizer.py` to set of cache directories that are assumed to exist. These should be
located in the relative directory `./cache/c_elegans`.
