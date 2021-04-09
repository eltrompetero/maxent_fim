# ====================================================================================== #
# Information geometry of social and biological networks.
# 
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from . import coarse_grain as cg
from . import fim as pyfim
from . import large_fim as lpyfim
from .organizer import MESolution, CoupSolution, MagSolution
from .organizer import CanonicalMagSolution, CanonicalCouplingSolution, FIM
from . import plot as pplt

from .spectral import *
from .utils import *
