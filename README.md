# Information geometry of maxent models

## Overview
Code to accompany preprint "Discovering sparse control strategies in C. elegans" 
(<http://arxiv.org/abs/2108.00837>) by Edward D. Lee, Xiaowen Chen and Bryan C. Daniels.

Jupyter notebook "plotting.ipynb" contains publication figures. The notebook generates
figures using calculations that have been pickled. At the moment, these must be requested
from the author.

## Additional notes
Computation for even modestly sized systems is expensive and best done on a large
computing cluster to leverage parallelization. Cached results from computations are
plotted in the accompanying Jupyter notebook. Code for HPC use is in another repository,
https://github.com/eltrompetero/unm_carc.

Given the number of calculations done, the code includes a wrapper defined in
`organizer.py` to set of cache directories that are assumed to exist. These should be
located in the relative directory `./cache/c_elegans`.


## Dependencies
The code depends on the following custom packages
- https://github.com/eltrompetero/misc
- https://github.com/eltrompetero/workspace
- https://github.com/eltrompetero/coniii
