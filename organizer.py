# ====================================================================================== #
# Module for organizing maxent solutions for neural data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import pickle
import os



class MESolution():
    DEFAULT_DR = 'cache/c_elegans'

    def __init__(self, name, ix):
        """
        Parameters
        ----------
        name : str
        ix : int
        """
        
        self.name = name
        self.ix = ix
        
        files = os.listdir(self.DEFAULT_DR)
        if not '%s%d.p'%(name, ix) in files:
            raise Exception("Maxent solution file not found.")
        self.n = pickle.load(open('%s/%s%d.p'%(self.DEFAULT_DR, name, ix), 'rb'))['X'].shape[1]
        if not '%s_fim%d.p'%(name, ix) in files:
            print("FIM not found.")

    def parameters(self):
        """Get maxent solution for specified data set.

        Returns
        -------
        ndarray
            Fields. All n * 3 of them.
        ndarray
            Couplings.
        """
        
        fname = '%s%d.p'%(self.name, self.ix)
        indata = pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))
        if 'h' in indata.keys() and 'J' in indata.keys():
            return indata['h'], indata['J']
        elif 'hJ' in indata.keys():
            return indata['hJ'][:3*self.n], indata['hJ'][3*self.n:]
        else:
            raise Exception("Could not find parameters in data pickle.")

    def fim(self):
        """
        Returns
        -------
        ndarray
        """

        fname = '%s_fim%d.p'%(self.fim, self.ix)
        return pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))['fim']
#end MESolution
