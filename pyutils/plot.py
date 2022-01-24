# ====================================================================================== #
# Plotting routines.
#
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from .utils import *
from .organizer import *



def plot_fit(n, sisjData, sisjModel,
             fig=None,
             ax=None,
             c='C0',
             draw_legend=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(11,4), ncols=2)
    
    ax[0].plot(sisjData[:n], sisjModel[:n], '.', zorder=1)
    ax[0].plot(sisjData[n:n*2], sisjModel[n:n*2], 'x', zorder=1)
    ax[0].plot(sisjData[n*2:n*3], sisjModel[n*2:n*3], '^', zorder=1)
    ax[0].set(xlabel=r'data $p(s_{\rm i}=k)$',
              ylabel=r'model $p(s_{\rm i}=k)$',
              xlim=[-.02,1.02], ylim=[-.02,1.02], yticks=[0,.5,1.])
    draw_one_to_one(ax[0])
    if draw_legend:
        ax[0].legend((r'$p(s_i=\rm rise)$',r'$p(s_i=\rm fall)$'),
                     handletextpad=.05,
                     borderpad=.2,
                     fontsize=20)

    ax[1].plot(sisjData[n*3:], sisjModel[n*3:], '.', c=c, zorder=1)
    ax[1].set(xlabel=r'data $p(s_{\rm i}=s_{\rm j})$',
              ylabel=r'model $p(s_{\rm i}=s_{\rm j})$',
              xlim=[-.02,1.02], ylim=[-.02,1.02], yticks=[0,.5,1.])
    if draw_legend:
        ax[1].legend((r'$p(s_i=s_j)$',),
                     handletextpad=.05,
                     borderpad=.2,
                     fontsize=20)
    draw_one_to_one(ax[1])

    fig.subplots_adjust(wspace=.5)
    return fig

def extract_pk(X):
    """Calculate phi_fine(k) from data.
    
    Parameters
    ----------
    X : ndarray
    
    Returns
    -------
    ndarray
    """
    
    Xk = np.zeros((len(X),3), dtype=int)  # number of neurons agreeing on state
    for k in range(3):
        Xk[:,k] = (X==k).sum(1)

    pk = np.bincount(Xk.max(1), minlength=X.shape[1]+1)
    pk = pk[int(np.ceil(X.shape[1]/3)):]
    
    pk = pk/pk.sum()
    return pk

def collective_fit(name, calc_observables):
    """Calculate statistics required to show goodness of fit to synchrony.
    
    Parameters
    ----------
    name : str
        Base name.
    calc_observables : function
        
    Returns
    -------
    ndarray
        Data sample.
    ndarray
        Data phi_fine.
    ndarray
        Indpt. model phi_fine.
    ndarray
        Pairwise maxent phi_fine.
    """
    
    soln = MESolution(name, 0)
    model = soln.model()
    X = soln.X()
    sisjData = calc_observables(X).mean(0)
    sisjModel = soln.sisj()
    n = soln.n

    from scipy.special import comb
    def multinomial(params):
        if len(params) == 1:
            return 1
        coeff = (comb(sum(params), params[-1], exact=True) *
                 multinomial(params[:-1]))
        return coeff

    def indpt_model_pk(N):
        pk = np.zeros(N - int(np.ceil(N/3)) + 1)
        for i, c in enumerate(range(int(np.ceil(N/3)), N+1)):
            for a in range(c+1):
                b = N - c - a
                if b<=c:
                    if (c==b and c!=a) or (c==a and c!=b):
                        pk[i] += multinomial([a, b, c]) / 2
                    elif (c==b and c==a):
                        pk[i] += multinomial([a, b, c]) / 6
                    else:
                        pk[i] += multinomial([a, b, c])
        pk /= pk.sum()
        return pk
    
    assert len(model.p)==len(model.allStates)
    
    return X, extract_pk(X), indpt_model_pk(X.shape[1]), extract_pk(model.allStates)

def grid4_h(ax, base_name, coarse_grain_type,
            mc='k',
            plot_kw={'ylim':(1e-4,2e2)},
            show_fit=False,
            show_detail=True,
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
    show_detail : bool, True
        If True, show each individual subsample instead of error bars.
    iprint : bool, False
    
    Returns
    -------
    list of handle
    """

    h = []

    for i, subset in enumerate(['A','B','C','D']):
        # plotting
        if iprint: print(f"Plotting subset {subset} corr pert.")
        soln = MagSolution(base_name, 0, 'a', 'i', subset,
                           coarse_grain_type=coarse_grain_type,
                           iprint=False)
        avgvals, vals = soln.avg_eigvals()
        if iprint and len(vals)<4: print("Missing some sims.")
        nonzeroix = avgvals>1e-7
        if show_detail:
            for v in vals:
                ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '.',
                             c=mc, alpha=.2, mew=0)
            h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '.',
                                  c=mc, mew=0)[0])
        else:
            h.append(ax[i].errorbar(range(1,nonzeroix.sum()+1), avgvals[nonzeroix],
                                    yerr=np.vstack(vals).std(0,ddof=1)[nonzeroix],
                                    fmt='.',
                                    c=mc, mew=0)[0])
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')


        if iprint: print(f"Plotting subset {subset} can pert.")
        soln = CanonicalMagSolution(base_name, 0, 'a', 'i', subset,
                                    coarse_grain_type=coarse_grain_type,
                                    iprint=False)
        avgvals, vals = soln.avg_eigvals()
        if iprint and len(vals)<4: print("Missing some sims.")
        nonzeroix = avgvals>1e-7
        if show_detail:
            for v in vals:
                ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '^',
                             c=mc, alpha=.2, mew=0)
            h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '^',
                                  c=mc, mew=0)[0])
        else:
            h.append(ax[i].errorbar(range(1,nonzeroix.sum()+1), avgvals[nonzeroix],
                                    yerr=np.vstack(vals).std(0,ddof=1)[nonzeroix],
                                    fmt='^',
                                    c=mc, mew=0)[0])
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')
 
        # labels and formatting
        if i==0:
            ax[i].set(ylabel='eigenvalue')
        elif i==1:
           pass
        elif i==2:
            ax[i].set(xlabel='rank', ylabel='eigenvalue')
        else:
            ax[i].set(xlabel='rank')
    ax[0].set(**plot_kw)

    return h

def grid4_J(ax, base_name, coarse_grain_type,
            mc='k',
            plot_kw={'ylim':(1e-4,2e2)},
            show_fit=False,
            show_detail=False,
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
    show_detail : bool, False
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
                            coarse_grain_type=coarse_grain_type,
                            iprint=False)
        try: 
            avgvals, vals = soln.avg_eigvals()
            if iprint and len(vals)<4: print("Missing some sims.")
            nonzeroix = avgvals>1e-7
            if show_detail:
                for v in vals:
                    ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '.',
                                 c=mc, alpha=.2, mew=0)
                h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '.',
                                      c=mc, mew=0)[0])
            else:
                h.append(ax[i].errorbar(range(1,nonzeroix.sum()+1), avgvals[nonzeroix],
                                        yerr=np.vstack(vals).std(0,ddof=1)[nonzeroix],
                                        fmt='.',
                                        c=mc, mew=0)[0])
        except ValueError:
            pass
        # power law fit
        if show_fit:
            y = avgvals[nonzeroix]
            x = np.arange(1, y.size+1)
            fit_fun = fit_decay_power_law(y, auto_upper_cutoff=-3.)[0]
            ax[i].loglog(x, fit_fun(x), 'k--')

        
        # plotting
        if iprint: print(f"Plotting subset {subset} can pert.")
        soln = CanonicalCouplingSolution(base_name, 0, 'a', 'i', subset,
                                         coarse_grain_type=coarse_grain_type,
                                         iprint=False)
        try: 
            avgvals, vals = soln.avg_eigvals()
            if iprint and len(vals)<4: print("Missing some sims.")
            nonzeroix = avgvals>1e-7
            if show_detail:
                for v in vals:
                    ax[i].loglog(range(1,nonzeroix.sum()+1), v[nonzeroix], '^',
                                 c=mc, alpha=.2, mew=0)
                h.append(ax[i].loglog(range(1,nonzeroix.sum()+1), avgvals[nonzeroix], '^',
                                      c=mc, mew=0)[0])
            else:
                h.append(ax[i].errorbar(range(1,nonzeroix.sum()+1), avgvals[nonzeroix],
                                        yerr=np.vstack(vals).std(0,ddof=1)[nonzeroix],
                                        fmt='^',
                                        c=mc, mew=0)[0])
        except ValueError:
            pass
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
