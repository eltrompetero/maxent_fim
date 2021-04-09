# ====================================================================================== #
# Module for organizing maxent solutions for neural data.
# 
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .large_fim import Mag3, Coupling3, CanonicalMag3, CanonicalCoupling3
from .spectral import sorted_eigh
from .utils import *



class MESolution():
    DEFAULT_DR = 'cache/c_elegans/coup3'  # default dr where pickles are stored
    # seconodary directory when using laptop setup, but hard drive must be connected
    if not os.path.isdir(DEFAULT_DR):
        DEFAULT_DR = "/Volumes/Eddie's SSD/Research/scotus4/py/cache/c_elegans"
        if not os.path.isdir(DEFAULT_DR):
            raise Exception("No cache found.")

    def __init__(self, name, data_ix,
                 soln_ix='a',
                 mc_ix='i',
                 subset_ix='A',
                 iprint=True,
                 coarse_grain_type=1):
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
        coarse_grain_type : int, 1
        """
        
        self.name = name
        self.subset_ix = subset_ix
        self.data_ix = data_ix
        self.soln_ix = soln_ix
        self.mc_ix = mc_ix
        self.ix = (subset_ix, str(data_ix), soln_ix, mc_ix)
        self.coarse_grain_type = coarse_grain_type
        
        self.DEFAULT_DR += f'/{coarse_grain_type}/'
        files = os.listdir(self.DEFAULT_DR)

        # look for maxent soln results
        if not self.soln_f() in files:
            if iprint: print(f"Maxent solution file {self.soln_f()} not found.")
            if not self.model_f() in files:
                raise Exception("Neither maxent nor model file found.")
            else:
                self.n = pickle.load(open(self.model_f(), 'rb'))['n']
                self.exists_model = True
            self._me = False
        else:
            fname = f'{self.DEFAULT_DR}/{self.soln_f()}'
            self.n = pickle.load(open(fname, 'rb'))['X'].shape[1]
            if not self.model_f() in files:
                if iprint: print(f"Model file {self.model_f()} not found.")
                self.exists_model = False
            else:
                self.exists_model = True
            self._me = True
        
        # look for FIM results
        if not self.fim_f() in files:
            if iprint: print(f"FIM file {self.fim_f()} not found.")
            self._fim = False
        else:
            self._fim = True

    def soln_f(self):
        return f'{self.name}_soln{"".join(self.ix[:-1])}.p'

    def model_f(self):
        return f"{self.name}_model{''.join(self.ix)}.p"

    def fim_f(self):
        return f"{self.name}_fim{''.join(self.ix)}.p"
        
    def neuron_ix(self):
        """Get indices of neurons solved from original discretized recording.

        Returns
        -------
        ndarray
        """
        
        if self._me:
            fname = self.soln_f()
        else:
            fname = self.model_f()
        indata = pickle.load(open(f'{self.DEFAULT_DR}/{fname}', 'rb'))
        
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
            fname = self.soln_f()
        else:
            fname = self.model_f()
        indata = pickle.load(open(f'{self.DEFAULT_DR}/{fname}', 'rb'))

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
    
    def setup_model(self, *, n_samples=100_000, eps=1e-4):
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
        
        model = Coupling3(self.n, h, J,
                          n_samples=n_samples,
                          eps=eps,
                          precompute=True,
                          coarse_grain_type=self.coarse_grain_type)
        pickle.dump(model.__get_state__(),
                    open(f'{self.DEFAULT_DR}/{self.model_f()}','wb'), -1)
        self.exists_model = True
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

        fname = f'{self.DEFAULT_DR}/{self.soln_f()}'
        if os.path.isfile(fname):
            return pickle.load(open(fname, 'rb'))['X']
        raise Exception(f"Solution file {fname} does not exist.")

    def model(self):
        if not self.exists_model:
            raise Exception("Model file not found.")

        if not '_model' in self.__dict__.keys():
            from .utils import load_Coupling3
            self._model = load_Coupling3(f'{self.DEFAULT_DR}/{self.model_f()}')
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
        """Return FIM from cache, else calculated it by calling spectral.sorted_eigh().

        Returns
        -------
        ndarray
        """

        fname = self.fim_f()
        indata = pickle.load(open(f'{self.DEFAULT_DR}/{fname}', 'rb'))
        fim = indata['fim']

        # automatically cache eigenvalues and eigenvectors for this fim
        if not 'eigvec' in indata.keys():
            indata['eigval'], indata['eigvec'] = sorted_eigh(fim)
            pickle.dump(indata, open(f'{self.DEFAULT_DR}/{fname}', 'wb'), -1)

        return fim

    def eig(self):
        """
        Returns
        -------
        ndarray
            Eigenvalues.
        ndarray
            Eigenvectors.
        """
        
        try:
            fname = self.fim_f()
            indata = pickle.load(open(f'{self.DEFAULT_DR}/{fname}', 'rb'))
            return indata['eigval'], indata['eigvec']
        except KeyError:
            self.fim()
            return self.eig()

    def avg_eigvals(self, return_eigvecs=False):
        """Rank-ordered eigenvalue spectrum averaged over MC samples used to calculate FIM.

        Parameters
        ----------
        return_eigvecs : bool, False

        Returns
        -------
        ndarray
            Average of sorted eigenvalue spectrum.
        ndarray
            All sorted eigenvalues by row.
        list of ndarray (optional)
            All eigenvectors.
        """
        
        #TODO: this is hard-coded, but should be flexible?
        rnumerals = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
        vals = []
        vecs = []

        # iterate through all available MC samples assuming that they are ordered consecutively
        for num in rnumerals:
            try:
                soln = self.__class__(self.name, self.data_ix,
                                      soln_ix=self.soln_ix, 
                                      mc_ix=num,
                                      subset_ix=self.subset_ix, 
                                      iprint=False,
                                      coarse_grain_type=self.coarse_grain_type)
                if soln._fim:
                    e = soln.eig()
                    vals.append(e[0])
                    vecs.append(e[1])
            except Exception:
                pass

        vals = np.vstack(vals)
        if return_eigvecs:
            return vals.mean(0), vals, vecs
        return vals.mean(0), vals

    def all_fim(self):
        """List of all FIM across MC samples.

        Returns
        -------
        list of ndarray
        """
        
        #TODO: this is hard-coded, but should be flexible
        rnumerals = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
        fim = []

        # iterate through all available MC samples assuming that they are ordered consecutively
        for num in rnumerals:
            try:
                soln = MESolution(self.name, self.data_ix,
                                  soln_ix=self.soln_ix, 
                                  mc_ix=num,
                                  subset_ix=self.subset_ix, 
                                  iprint=False)
                if soln._fim:
                    fim.append(soln.fim())
            except Exception:
                pass

        return fim
#end MESolution

CoupSolution = MESolution



class MagSolution(MESolution):
    DEFAULT_DR = 'cache/c_elegans/mag3'  # default dr where pickles are stored

    def setup_model(self, *, n_samples=100_000, eps=1e-4):
        """Set up large_fim.Mag3 for FIM calculation. Result will be saved into model
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
        
        model = Mag3(self.n, h, J,
                     n_samples=n_samples,
                     eps=eps,
                     precompute=True,
                     coarse_grain_type=self.coarse_grain_type)
        pickle.dump(model.__get_state__(),
                    open(f'{self.DEFAULT_DR}/{self.model_f()}','wb'), -1)
        self.exists_model = True
        # this pickle can be loaded using utils.load_Coupling3 as shown in self.model()

    def setup_fim(self):
        """Calculate and pickle FIM."""
        
        model = self.model()
        fname = self.fim_f()
        fim = model.maj_curvature()

        pickle.dump({'fim':fim}, open(f'{self.DEFAULT_DR}/{fname}', 'wb'), -1)

    def model(self):
        if not self.exists_model:
            raise Exception("Model file not found.")

        if not '_model' in self.__dict__.keys():
            from .utils import load_Mag3
            self._model = load_Mag3(f'{self.DEFAULT_DR}/{self.model_f()}')
        return self._model
#end MagSolution



class CanonicalMagSolution(MESolution):
    DEFAULT_DR = 'cache/c_elegans/can_mag3'  # default dr where pickles are stored

    def setup_model(self, *, n_samples=100_000, eps=1e-4):
        """Set up large_fim.Mag3 for FIM calculation. Result will be saved into model
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
        
        model = CanonicalMag3(self.n, h, J,
                              n_samples=n_samples,
                              eps=eps,
                              precompute=True,
                              coarse_grain_type=self.coarse_grain_type)
        pickle.dump(model.__get_state__(),
                    open(f'{self.DEFAULT_DR}/{self.model_f()}','wb'), -1)
        self.exists_model = True

    def setup_fim(self):
        """Calculate and pickle FIM."""
        
        model = self.model()
        fname = self.fim_f()
        fim = model.maj_curvature()

        pickle.dump({'fim':fim}, open(f'{self.DEFAULT_DR}/{fname}', 'wb'), -1)

    def model(self):
        if not self.exists_model:
            raise Exception("Model file not found.")

        if not '_model' in self.__dict__.keys():
            from .utils import load_CanonicalMag3
            self._model = load_CanonicalMag3(f'{self.DEFAULT_DR}/{self.model_f()}')
        return self._model
#end CanonicalMagSolution



class CanonicalCouplingSolution(MESolution):
    DEFAULT_DR = 'cache/c_elegans/can_coup3'  # default dr where pickles are stored

    def setup_model(self, *, n_samples=100_000, eps=1e-4):
        """Set up large_fim.CanonicalCoupling3 for FIM calculation. Result will be saved
        into model pickle.

        Parameters
        ----------
        n_samples : int, 1e5
        eps : float, 1e-4

        Returns
        -------
        None
        """

        h, J = self.parameters()
        
        model = CanonicalCoupling3(self.n, h, J,
                                   n_samples=n_samples,
                                   eps=eps,
                                   precompute=True,
                                   coarse_grain_type=self.coarse_grain_type)
        pickle.dump(model.__get_state__(),
                    open(f'{self.DEFAULT_DR}/{self.model_f()}','wb'), -1)
        self.exists_model = True

    def setup_fim(self):
        """Calculate and pickle FIM."""
        
        model = self.model()
        fname = self.fim_f()
        fim = model.maj_curvature()

        pickle.dump({'fim':fim}, open(f'{self.DEFAULT_DR}/{fname}', 'wb'), -1)

    def model(self):
        if not self.exists_model:
            raise Exception("Model file not found.")

        if not '_model' in self.__dict__.keys():
            from .utils import load_CanonicalCoupling3
            self._model = load_CanonicalCoupling3(f'{self.DEFAULT_DR}/{self.model_f()}')
        return self._model
#end CanonicalCouplingSolution



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
        from scipy.special import binom

        # check that given number of samples does not exceed max number of possible subsets
        assert n_sample <= int(binom(self.n, n_comp))

        topval = np.zeros(n_sample)

        for i in range(n_sample):
            val, vec = subspace_eig(self.fim, np.random.choice(range(self.n),
                                                               size=n_comp,
                                                               replace=False))
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
            TODO: do not repeat subsets
        """
        
        max_subset_size = max_subset_size or self.n
        assert max_subset_size>0
        
        sampleVal = []
        if n_subset_range is None:  # default range
            n_subset_range = list(range(1, self.n+1, 5))
        if type(n_subset_range) is int:  # space range out automatically
            n_subset_range = list(range(1, self.n+1, n_subset_range))

        for nComp in n_subset_range:
            sampleVal.append( self._sample_subset_eigval(nComp, min(int(binom(self.n,nComp)),max_subset_size)) )
        
        return sampleVal

    def eig(self, tol=1e-7):
        """Wrapper for np.linalg.eig that takes the real part, throws away small
        eigenvalue components, and sorts the results by eigenvalue.
        
        Parameters
        ----------
        tol : float, 1e-7
        
        Returns
        -------
        ndarray
            Eigenvalues.
        ndarray
            Eigenvectors by col.
        """

        val, vec = np.linalg.eigh(self.fim)
        
        nonzeroix = val.real>tol
        val = val[nonzeroix]
        vec = vec[:,nonzeroix]

        sortix = np.argsort(val.real)[::-1]
        val = val[sortix].real
        vec = vec[:,sortix].real

        return val, vec
#end MESolution
