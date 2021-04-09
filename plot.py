# ====================================================================================== #
# Plotting routines.
#
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from .organizer import *



def grid4_h(ax, base_name, coarse_grain_type,
            mc='k',
            plot_kw={'ylim':(1e-4,2e2)},
            show_fit=False):
    """
    Parameters
    ----------
    ax : list of plt.Axes
    base_name : str
    coarse_grain_type : int
    mc : str, 'k'
    plot_kw : dict, {'ylim':(1e-4,2e2)}
    show_fit : bool, False
    
    Returns
    -------
    list of handle
    """

    h = []

    for i, subset in enumerate(['A','B','C','D']):
        # plotting
        soln = MagSolution(base_name, 0, 'a', 'i', subset,
                           coarse_grain_type=coarse_grain_type)
        avgvals, vals = soln.avg_eigvals()
        nonzeroix = avgvals>1e-7
        for v in vals:
            ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '.',
                         c=mc, alpha=.2, mew=0)
        h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '.',
                              c=mc, mew=0)[0])
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')
 
        soln = CanonicalMagSolution(base_name, 0, 'a', 'i', subset,
                                    coarse_grain_type=coarse_grain_type)
        avgvals, vals = soln.avg_eigvals()
        nonzeroix = avgvals>1e-7
        for v in vals:
            ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '^',
                         c=mc, alpha=.2, mew=0)
        h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '^',
                              c=mc, mew=0)[0])
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')
 
        # labels and formatting
        if i==0:
            ax[i].set(ylabel='eigenval')
        elif i==1:
           pass
        elif i==2:
            ax[i].set(xlabel='eigenval rank', ylabel='eigenval')
        else:
            ax[i].set(xlabel='eigenval rank')
    ax[0].set(**plot_kw)

    return h

def grid4_J(ax, base_name, coarse_grain_type,
            mc='k',
            plot_kw={'ylim':(1e-4,2e2)},
            show_fit=False,
            iprint=False):
    """
    Parameters
    ----------
    ax : list of plt.Axes
    base_name : str
    coarse_grain_type : int
    mc : str, 'k'
    plot_kw : dict, {'ylim':(1e-4,2e2)}
    show_fit : bool, False
    iprint : bool, False
    
    Returns
    -------
    list of handles
    """
    
    h = []

    for i, subset in enumerate(['A','B','C','D']):
        # plotting
        if iprint: print(f"Plotting subset {subset} corr pert.")
        soln = CoupSolution(base_name, 0, 'a', 'i', subset,
                            coarse_grain_type=coarse_grain_type)
        avgvals, vals = soln.avg_eigvals()
        nonzeroix = avgvals>1e-7
        for v in vals:
            ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '.',
                         c=mc, alpha=.2, mew=0)
        h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '.',
                              c=mc, mew=0)[0])
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')
        
        # plotting
        if iprint: print(f"Plotting subset {subset} can pert.")
        soln = CanonicalCouplingSolution(base_name, 0, 'a', 'i', subset,
                                         coarse_grain_type=coarse_grain_type)
        avgvals, vals = soln.avg_eigvals()
        nonzeroix = avgvals>1e-7
        for v in vals:
            ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '^',
                         c=mc, alpha=.2, mew=0)
        h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '^',
                              c=mc, mew=0)[0])
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')
        
        # labels and formatting
        if i==0:
            ax[i].set(ylabel='eigenval')
        elif i==1:
           pass
        elif i==2:
            ax[i].set(xlabel='eigenval rank', ylabel='eigenval')
        else:
            ax[i].set(xlabel='eigenval rank')
    ax[0].set(**plot_kw)

    return h
