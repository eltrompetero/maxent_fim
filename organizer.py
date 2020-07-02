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
            if not f'{name}_model{"".join(self.ix)}.p' in files:
                raise Exception("Neither maxent nor model file found.")
            else:
                self.n = pickle.load(open(f'{self.DEFAULT_DR}/{name}_model{"".join(self.ix)}.p', 'rb'))['n']
                self.exists_model = True
            self._me = False
        else:
            fname = f'{self.DEFAULT_DR}/{name}_soln{"".join(self.ix[:-1])}.p'
            self.n = pickle.load(open(fname, 'rb'))['X'].shape[1]
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

    def sisj(self, source='model'):
        """Observables (means and pairwise correlations) calculated from model.
        
        Parameters
        ----------
        source : str, 'model'
            'model' or 'data'

        Returns
        -------
        ndarray
            Means.
        ndarray
            Pairwise correlations.
        """

        if not '_model' in self.__dict__.keys():
            self.model()

        if source=='model':
            return self._model.sisj[:150], self._model.sisj[150:]

        calc_observables = define_potts_helper_functions(3)[1]
        sisj = calc_observables(self.X()).mean(0)
        return sisj[:150], sisj[150:]

    def fim(self):
        """
        Returns
        -------
        ndarray
        """

        fname = '%s_fim%s.p'%(self.name, ''.join(self.ix))
        return pickle.load(open('%s/%s'%(self.DEFAULT_DR, fname), 'rb'))['fim']

    def avg_eigvals(self):
        """Rank ordered eigenvalue spectrum averaged over MC samples used to calculate FIM.

        Returns
        -------
        ndarray
            Average of sorted eigenvalue spectrum.
        ndarray
            All sorted eigenvalues by row.
        """
        
        rnumerals = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
        vals = []

        # iterate through all available MC samples assuming that they are ordered consecutively
        counter = 0
        fname = f'{self.DEFAULT_DR}/{self.name}_fim{"".join(self.ix[:-1])}{rnumerals[counter]}.p'
        while os.path.isfile(fname):
            fim = pickle.load(open(fname, 'rb'))['fim']

            v, vec = self.model().hess_eig(fim, iprint=False)
            vals.append(v)

            counter += 1
            fname = f'{self.DEFAULT_DR}/{self.name}_fim{"".join(self.ix[:-1])}{rnumerals[counter]}.p'

            assert counter<10

        vals = np.vstack(vals)
        return vals.mean(0), vals
#end MESolution



class FIM():
    """Class for helping organize types of calculation to be done on FIM.
    """
    def __init__(self, n, fim):
        self.n = n
        self.fim = fim
        # perhaps include option to store fim on disk to save memory

    def _sample_subset_eigval(self, n_comp, n_sample):
        """
        Parameters
        ----------
        n_comp : int
            Number of components to sample.
        n_sample : int
            Number of subsets to try.

        Returns
        -------
        ndarray
        """
        
        from .influence import subspace_eig
        topval = np.zeros(n_sample)

        for i in range(n_sample):
            val, vec = subspace_eig(self.fim, np.random.choice(range(self.n),
                                                               size=n_comp, replace=False))
            topval[i] = val[0]

        return topval
    
    def sample_subset_eigval(self, n_subset_range=None, max_subset_size=None):
        """Calculate eigenvalue spectrum for subspace spanned by random groups of
        n_comp components.
        
        Parameters
        ----------
        n_subset_range : ndarray, None
        max_subset_size : int, 50
        
        Returns
        -------
        list of ndarray
            Each ndarray contains top eigenvalue from multiple random subsets. Subsets might repeat.
        """
        
        max_subset_size = max_subset_size or self.n
        assert max_subset_size>0
        
        sampleVal = []
        if n_subset_range is None:
            n_subset_range = list(range(1, self.n+1, 5))
        if type(n_subset_range) is int:
            n_subset_range = list(range(1, self.n+1, n_subset_range))

        for nComp in n_subset_range:
            sampleVal.append( self._sample_subset_eigval(nComp, min(int(binom(self.n,nComp)),max_subset_size)) )
        
        return sampleVal
#end FIM
