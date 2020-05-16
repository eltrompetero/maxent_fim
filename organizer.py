# ====================================================================================== #
# Module for organizing maxent solutions for neural data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import pickle
import os



class MESolution():
    DEFAULT_DR = 'cache/c_elegans'

    def __init__(self, name, ix, iprint=True):
        """
        Parameters
        ----------
        name : str
        ix : int
        """
        
        self.name = name
        self.ix = ix
        
        files = os.listdir(self.DEFAULT_DR)

        # look for maxent soln results
        if not '%s_soln%d.p'%(name, ix) in files:
            if iprint: print("Maxent solution file not found.")
            if not '%s_model%d.p'%(name, ix) in files:
                raise Exception("Neither maxent nor model file found.")
            else:
                self.n = pickle.load(open('%s/%s_model%d.p'%(self.DEFAULT_DR, name, ix), 'rb'))['n']
                self._model = True
            self._me = False
        else:
            self.n = pickle.load(open('%s/%s_soln%d.p'%(self.DEFAULT_DR, name, ix), 'rb'))['X'].shape[1]
            if not '%s_model%d.p'%(name, ix) in files:
                if iprint: print("Model file not found.")
                self._model = False
            else:
                self._model = True
            self._me = True
        
        # look for FIM results
        if not '%s_fim%d.p'%(name, ix) in files:
            if iprint: print("FIM not found.")
            self._fim = False
        else:
            self._fim = True

    def parameters(self):
        """Get maxent solution for specified data set.

        Returns
        -------
        ndarray
            Fields. All n * 3 of them.
        ndarray
            Couplings.
        """
        
        if self._me:
            fname = '%s_soln%d.p'%(self.name, self.ix)
        else:
            fname = '%s_model%d.p'%(self.name, self.ix)
        indata = pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))

        if 'h' in indata.keys() and 'J' in indata.keys():
            return indata['h'], indata['J']
        elif 'hJ' in indata.keys():
            return indata['hJ'][:3*self.n], indata['hJ'][3*self.n:]
        else:
            raise Exception("Could not find parameters in data pickle.")
    
    def model(self):
        from .utils import load_Coupling3
        return load_Coupling3('%s/%s_model%d.p'%(self.DEFAULT_DR, self.name, self.ix))

    def fim(self):
        """
        Returns
        -------
        ndarray
        """

        fname = '%s_fim%d.p'%(self.name, self.ix)
        return pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))['fim']
#end MESolution
