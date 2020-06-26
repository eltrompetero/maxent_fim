# ====================================================================================== #
# Module for organizing maxent solutions for neural data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import os
from .large_fim import Coupling3
from .utils import *
import dill



class MESolution():
    DEFAULT_DR = 'cache/c_elegans'  # default dr where pickles are stored
    #DEFAULT_DR = "/Volumes/Eddie's SSD/Research/scotus4/py/cache/c_elegans"  # temp dr where pickles are stored

    def __init__(self, name, data_ix,
                 soln_ix='a',
                 mc_ix='i',
                 subset_ix='A',
                 iprint=True):
        """
        Parameters
        ----------
        name : str
        data_ix : int
            Sample index for a given data set.
        soln_ix : 'a'
            Solution method index.
        mc_ix : 'i'
            Monte Carlo sample index.
        subset_ix : str, 'A'
            Subset index of neurons.
        """
        
        self.name = name
        self.subset_ix = subset_ix
        self.data_ix = data_ix
        self.soln_ix = soln_ix
        self.mc_ix = mc_ix
        self.ix = (subset_ix, str(data_ix), soln_ix, mc_ix)
        
        files = os.listdir(self.DEFAULT_DR)

        # look for maxent soln results
        if not f'{name}_soln{"".join(self.ix[:-1])}.p' in files:
            if iprint: print("Maxent solution file not found.")
            if not '%s_model%s.p'%(name, ''.join(self.ix)) in files:
                raise Exception("Neither maxent nor model file found.")
            else:
                self.n = pickle.load(open(f'{self.DEFAULT_DR}/{name}_model{"".join(self.ix)}.p', 'rb'))['n']
                self.exists_model = True
            self._me = False
        else:
            self.n = pickle.load(open('%s/%s_soln%s.p'%(self.DEFAULT_DR,
                                                        name,
                                                        ''.join(self.ix[:-1])), 'rb'))['X'].shape[1]
            if not '%s_model%s.p'%(name, ''.join(self.ix)) in files:
                if iprint: print("Model file not found.")
                self.exists_model = False
            else:
                self.exists_model = True
            self._me = True
        
        # look for FIM results
        if not '%s_fim%s.p'%(name, ''.join(self.ix)) in files:
            if iprint: print("FIM not found.")
            self._fim = False
        else:
            self._fim = True

    def neuron_ix(self):
        """Get indices of neurons solved from original discretized recording.

        Returns
        -------
        ndarray
        """
        
        if self._me:
            fname = '%s_soln%s.p'%(self.name, ''.join(self.ix[:-1]))
        else:
            fname = '%s_model%s.p'%(self.name, ''.join(self.ix[:-1]))
        indata = pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))
        
        return indata['neuronix']

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
            fname = '%s_soln%s.p'%(self.name, ''.join(self.ix[:-1]))
        else:
            fname = '%s_model%s.p'%(self.name, ''.join(self.ix[:-1]))
        indata = pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))

        if 'h' in indata.keys() and 'J' in indata.keys():
            if indata['h'].size==2*self.n:
                return np.concatenate((indata['h'], np.zeros(self.n))), indata['J']
            return indata['h'], indata['J']
        elif 'hJ' in indata.keys():
            if indata['hJ'].size==(self.n*3 + self.n*(self.n-1)//2):
                return indata['hJ'][:3*self.n], indata['hJ'][3*self.n:]
            return np.concatenate((indata['hJ'][:2*self.n], np.zeros(self.n))), indata['hJ'][2*self.n:]
        else:
            raise Exception("Could not find parameters in data pickle.")
    
    def setup_model(self, n_samples=100_000, eps=1e-4):
        """Set up large_fim.Coupling3 for FIM calculation. Result will be saved into model
        pickle.

        Parameters
        ----------
        n_samples : int, 1e5
        eps : float, 1e-4

        Returns
        -------
        None
        """

        h, J = self.parameters()

        model = Coupling3(self.n, h, J, n_samples=n_samples, eps=eps, precompute=True)
        dill.dump(model.__get_state__(),
                  open('cache/c_elegans/%s_model%s.p'%(self.name, ''.join(self.ix)),'wb'), -1)
        # this pickle can be loaded using utils.load_Coupling3 as shown in self.model()
    
    def X(self):
        """Return sample on which model was solved.

        Parameters
        ----------

        Returns
        -------
        ndarray
            Dimensions of (n_samples, n_neurons).
        """

        fname = f"{self.DEFAULT_DR}/{self.name}_soln{''.join(self.ix[:-1])}.p"
        if os.path.isfile(fname):
            return pickle.load(open(fname, 'rb'))['X']
        raise Exception(f"Solution file {self.DEFAULT_DR}/{self.name}_soln{''.join(self.ix[:-1])}.p does not exist.")

    def model(self):
        if not self.exists_model:
            raise Exception("Model file not found.")

        if not '_model' in self.__dict__.keys():
            from .utils import load_Coupling3
            self._model = load_Coupling3('%s/%s_model%s.p'%(self.DEFAULT_DR, self.name, ''.join(self.ix)))
        return self._model

    def sisj(self):
        if not '_model' in self.__dict__.keys():
            self.model();
        return self._model.sisj

    def fim(self):
        """
        Returns
        -------
        ndarray
        """

        fname = '%s_fim%s.p'%(self.name, ''.join(self.ix))
        return pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))['fim']
#end MESolution
