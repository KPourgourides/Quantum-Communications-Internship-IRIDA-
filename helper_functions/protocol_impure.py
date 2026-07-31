import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import math
from scipy.optimize import curve_fit, brentq
from helper_functions.state_measurement import *
from scipy.special import erfc
import plotly.graph_objects as go
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d


# CONTENTS:
#==================================
# ERROR PROBABILITY
# BETA THRESHOLD & OPTIMAL SQUEEZING
# SIGMA THRESHOLD 
# HELSTROM

# Dataset selection
global DATAPATH_DTS, DATAPATH_DSTS
DATAPATH_DTS = 'data/DTS/perr_dts_mu201_S1000000000'
DATAPATH_DSTS = 'data/DSTS/perr_dsts_b402_mu201_S1000000000'

#======================================================================================
#                           ERROR PROBABILITY
#======================================================================================

def perr_dsts(N:float, mu_grid:np.array, beta_grid:np.array, sigma:float, num_samples:int, strawberry:bool=False) -> np.array:
        '''
        Experimental calculation of the homodyne error probability for displaced squeezed thermal states
        '''
        p_err = np.full((len(mu_grid), len(beta_grid)), np.nan)
        mu_min = 1/(1+2*N)

        #Calculation of error probability
        #============================================        
        if not strawberry:

            gauss = hermgauss(100)

            for k, mu in enumerate(mu_grid):
                beta_max = mu + (mu - 1)/(2*N)
                print(f"\rProgress: {k+1}/{len(mu_grid)}", end="", flush=True)

                if mu >= mu_min:
                     
                    for i, beta in enumerate(beta_grid):
                        if beta <= beta_max:
                            p = theory_point_dsts(N, beta, mu, sigma, gauss)
                            p_err[k, i] = np.random.binomial(num_samples, p) / num_samples
        #============================================   
        else:

            eng = sf.Engine("gaussian")
        
            for i, mu in enumerate(mu_grid):
                if mu >= mu_min:

                    alphas2 = N*(1-beta_grid)+(mu-1)*(1+2*N*beta_grid)/(2*mu)
                    alpha_valid_indices = np.where(alphas2 >= 0)[0]
                    alphas = np.sqrt(alphas2[alpha_valid_indices])
                    Nth = (1 - mu)/(2*mu)
                
                    print(f"\rProgress: {i+1}/{len(mu_grid)}", end="", flush=True)

                    for local_idx, global_idx in enumerate(alpha_valid_indices):
        
                            wrong_sign_counter = 0
                            r_s = math.asinh(np.sqrt(N*beta_grid[global_idx]))

                            # Choose phase from Gaussian distribution
                            #============================================
                            phis = np.random.normal(0, sigma, size = num_samples)

                            for phi in phis:

                                coherent_sign = np.random.choice([1, -1])
                                prog = sf.Program(1)

                                with prog.context as q:
                                    Thermal(Nth) | q
                                    Sgate(r_s, 0) | q[0] 
                                    Dgate(coherent_sign*alphas[local_idx]) | q
                                    Rgate(phi) | q
                                    MeasureHomodyne(0) | q

                                result = eng.run(prog)
                                eng.reset()
                                result_sign = np.sign(result.samples[0][0])

                                if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                                    wrong_sign_counter+= 1

                            p_err[i][global_idx] = wrong_sign_counter/num_samples

        return p_err


def perr_dts(N:float, mu_grid:np.array, sigma:float, num_samples:int, strawberry:bool=False) -> np.array:
        '''
        Experimental calculation of the homodyne error probability for coherent states
        '''

        p_err = np.full((len(mu_grid)), np.nan)
        mu_min = 1/(1+2*N)

        #Calculation of error probability
        #============================================
        if not strawberry:
             
            gauss = hermgauss(100)

            for i, mu in enumerate(mu_grid):

                if mu >= mu_min:

                    print(f"\rProgress: {i+1}/{len(mu_grid)}", end="", flush=True)

                    p = theory_point_dts(N, mu, sigma, gauss)
                    p_err[i] = np.random.binomial(num_samples, p) / num_samples
        #============================================
        else:
             
            eng = sf.Engine("gaussian")
            for i, mu in enumerate(mu_grid):

                if mu >= mu_min:

                    alpha = np.sqrt(N+(mu-1)/(2*mu))
                    Nth = (1 - mu)/(2*mu)

                    print(f"\rProgress: {i+1}/{len(mu_grid)}", end="", flush=True)

                    wrong_sign_counter = 0

                    # Choose phase from Gaussian distribution
                    #============================================

                    phis = np.random.normal(0, sigma, size = num_samples)

                    for phi in phis:

                        coherent_sign = np.random.choice([1, -1])
                        prog = sf.Program(1)

                        with prog.context as q:
                            Thermal(Nth) | q
                            Dgate(coherent_sign*alpha) | q
                            Rgate(phi) | q
                            MeasureHomodyne(0) | q

                        result = eng.run(prog)
                        eng.reset()
                        result_sign = np.sign(result.samples[0][0])
                        if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                            wrong_sign_counter+= 1

                    p_err[i] = wrong_sign_counter/num_samples

        return p_err


def theory_point_dsts(N:float, beta:float, mu:float, sigma:float, gauss:tuple) -> float:
        '''
        Numerical calculation of the theoretical homodyne error probability for displaced squeezed thermal states,
        works for a single (N, β, μ, σ) points.
        '''
        x_gh, w_gh = gauss

        a2 = N*(1-beta)+(mu-1)*(1+2*N*beta)/(2*mu)
        a = np.sqrt(np.clip(a2, 0, None))
        r = np.arcsinh(np.sqrt(N * beta))
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r) * np.cos(phi)**2 + np.exp(2*r) * np.sin(phi)**2)/mu

        arg = np.sqrt(2) * a * np.cos(phi)/np.sqrt(V)
        integral = 0.5*np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return integral

def theory_point_dts(N:float, mu:float, sigma:float, gauss:tuple) -> float:
        '''
        Numerical calculation of the theoretical homodyne error probability for displaced thermal states,
        works for a single (N, σ, μ) points.
        '''
        x_gh, w_gh = gauss

        a = np.sqrt(N+(mu-1)/(2*mu))
        r = 0
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r) * np.cos(phi)**2 + np.exp(2*r) * np.sin(phi)**2)/mu

        arg = np.sqrt(2) * a * np.cos(phi)/np.sqrt(V)
        integral = 0.5*np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return integral


def plot_homodyne_perr(N:float, sigmas:list, colors_light:list, colors_dark:list, dts:str|bool = False, dsts:str|bool = False) -> None:
    '''
    Plots the homodyne error probability for dts and dsts.
    dts and dsts can only take the values 'data', 'theory', 'all', or False
    '''
    
    valid_args = {'data', 'theory', 'all', False}

    if dts not in valid_args or dsts not in valid_args:
         raise KeyError("dts and dsts can only take the values 'data', 'theory', 'all', or False")

    n_gh = 100
    gauss = hermgauss(n_gh)
    
    fig = go.Figure()
    
    #============================  dts  ============================
    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_dts = np.load(f"{DATAPATH_DTS}_N{N}_sigma{sigma}.npz")

        N_dts = data_dts["N"]
        perr_dts =  data_dts["perr"]
        mus_dts = data_dts["mus_grid"]
        beta_dts = np.linspace(0, 1, len(mus_dts))
        
        N_surface_dts, beta_surface_dts = np.meshgrid(mus_dts, beta_dts, indexing="ij")
        z_surface_dts = np.zeros_like(N_surface_dts)
        perr_surface_dts = np.zeros_like(N_surface_dts)

        mu_min = 1/(1+2*N_dts)
        #-------------------------  Theoretical curve  -------------------------
        
        for k in range(len(mus_dts)):
            if mus_dts[k] >= mu_min:
                for l in range(len(beta_dts)):
                        perr_surface_dts [k, l] = perr_dts[k]
                        z_surface_dts[k, l] = theory_point_dts(N_dts, mus_dts[k], sigma, gauss)
        
        #-------------------------  R^2  -------------------------
        
        ss_res_dts = np.sum((perr_dts - z_surface_dts[:,0])**2)
        ss_tot_dts = np.sum((perr_dts - np.mean(perr_dts))**2)
        R2_dts = 1 - ss_res_dts/ss_tot_dts
        
        #-------------------------  Plot  -------------------------
        
        if dts in ['theory', 'all']:
        
            fig.add_trace(go.Surface(x=N_surface_dts, y=beta_surface_dts, z=z_surface_dts, surfacecolor=np.zeros_like(z_surface_dts), 
                                colorscale=[[0.0, colors_light[i]], [1.0, colors_light[i]]], showscale=False))
        
        if dts in ['data', 'all']:
            
            fig.add_trace(go.Scatter3d(x=N_surface_dts.ravel(), y=beta_surface_dts.ravel(), z=perr_surface_dts.ravel(), mode="markers", 
                                marker=dict(size=3, color=colors_dark[i])))

            

    
    #============================  dsts  ============================

    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_dsts = np.load(f"{DATAPATH_DSTS}_N{N}_sigma{sigma}.npz")
        N_dsts =  data_dsts["N"]
        beta_dsts =  data_dsts["beta_grid"]
        perr_dsts = data_dsts["perr"]
        mus_dsts = data_dsts["mus_grid"]

        mu_surface_dsts, beta_surface_dsts = np.meshgrid(mus_dsts, beta_dsts, indexing="ij")
        
        #-------------------------  Theoretical curve  -------------------------
        
        z_surface_dsts = np.zeros_like(mu_surface_dsts)

        for k in range(len(mus_dsts)):
            beta_max = mus_dsts[k] + (mus_dsts[k] - 1)/(2*N_dsts)
            for l in range(len(beta_dsts)):
                if beta_dsts[l] <= beta_max:
                    z_surface_dsts[k, l] = theory_point_dsts(N_dsts, beta_dsts[l], mus_dsts[k], sigma, gauss)
        
        #-------------------------  R^2  -------------------------
        
        ss_res_dsts = np.sum((perr_dsts - z_surface_dsts)**2)
        ss_tot_dsts = np.sum((perr_dsts - np.mean(perr_dsts))**2)
        R2_dsts = 1 - ss_res_dsts/ss_tot_dsts
        
        #-------------------------  Plot  -------------------------
        
        if dsts in ['theory', 'all']:
                fig.add_trace(go.Surface( x=mu_surface_dsts, y=beta_surface_dsts, z=z_surface_dsts, surfacecolor=np.zeros_like(z_surface_dsts), 
                                colorscale=[[0.0, colors_dark[i]], [1.0, colors_dark[i]]], showscale=False))
        
        if dsts in ['data', 'all']:
                fig.add_trace(go.Scatter3d(x=mu_surface_dsts.ravel(), y=beta_surface_dsts.ravel(), z=perr_dsts.ravel(),
                mode="markers", marker=dict(size=3, color=colors_light[i])))
                

        fig.update_layout(scene=dict(xaxis_title="μ", yaxis_title=r"β", zaxis = dict(title="P_err", type="log"), 
                            aspectmode ="cube"), width=900, height=750)
        
    if dts or dsts:
        fig.show()

#======================================================================================
#                        BETA THRESHOLD & OPTIMAL SQUEEZING
#======================================================================================

def beta_threshold_theory(N: float, mu: float, sigma: float, gauss: tuple) -> float:
    """
    Finds the threshold squeezing fraction β_th for fixed N, μ and σ by solving

        P_err,dsts(β) = P_err,dts

    over the physically allowed interval.
    """

    point_dts = theory_point_dts(N, mu, sigma, gauss)

    def F(beta):
        return theory_point_dsts(N, beta, mu, sigma, gauss) - point_dts

    # ---------- Physical upper bound ----------
    beta_phys_max = mu + (mu - 1) / (2 * N)

    # No physical dsts exists
    if beta_phys_max <= 0:
        return np.nan

    # ---------- Locate the minimum of F ----------
    beta_grid = np.linspace(0, beta_phys_max, 500)
    F_grid = np.array([F(b) for b in beta_grid])

    beta_min = beta_grid[np.argmin(F_grid)]

    # ---------- Find the second intersection ----------
    try:
        return brentq(F, beta_min, beta_phys_max)

    except ValueError:
        print(f"No root for mu = {mu:.4f}")
        return np.nan
    

def beta_optimal_theory(N:float, mu:float, sigma:float, gauss:tuple) -> float:
    '''
    Utilized an optimization method to find the value of β in [0,1] which minimizes the homodyne error probability;
    thus, the optimal value of β.
    '''
    def objective(beta:float):
        '''
        Defined the objective function for the minimization, which is the homodyne error probability
        '''
        return theory_point_dsts(N, beta, mu, sigma, gauss)

    beta_max = mu + (mu - 1)/(2*N)
    if beta_max <= 0:
        return np.nan

    res = minimize_scalar(objective, bounds=(0, beta_max), method="bounded")
    return res.x


def beta_optimal_data(N, sigmas:list) -> dict:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for different sigmas.
    '''
    #---------- FIND THRESHOLD ----------
    beta_opt_dict = {}

    for i,sigma in enumerate(sigmas):

        # ------------- LOAD DATA ----------------------
               
        data_dts = np.load(f"{DATAPATH_DTS}_N{N}_sigma{sigma}.npz")
        perr_dts =  data_dts["perr"]

        data_dsts = np.load(f"{DATAPATH_DSTS}_N{N}_sigma{sigma}.npz")
        perr_dsts =  data_dsts["perr"]
        beta = data_dsts["beta_grid"]
        mus_grid = data_dsts["mus_grid"]
        N = data_dsts["N"]
    
        mus_surface, beta_surface = np.meshgrid(mus_grid, beta, indexing="ij")
        perr_surface_dts = np.zeros_like(mus_surface)

        for k in range(len(mus_grid)):
            for l in range(len(beta)):

                perr_surface_dts [k, l] = perr_dts[k]
        
        # ------------- OPTIMAL ----------------------

        # Minima along beta for each N
        valid_rows = ~np.all(np.isnan(perr_dsts), axis=1)
        beta_opt = np.full(len(mus_grid), np.nan)
        idx = np.nanargmin(perr_dsts[valid_rows], axis=1)
        beta_opt[valid_rows] = beta[idx]
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt
        
    return beta_opt_dict

def beta_threshold_data(N: float, sigmas: list) -> dict:
    """
    Finds the threshold β_th by locating the zero crossing of
    perr_dts - perr_dsts along the beta axis for each μ.

    NaN values are ignored independently for each μ.
    """

    beta_th_dict = {}

    for sigma in sigmas:

        # ---------- Load data ----------
        data_dts = np.load(f"{DATAPATH_DTS}_N{N}_sigma{sigma}.npz")
        perr_dts = data_dts["perr"]

        data_dsts = np.load(f"{DATAPATH_DSTS}_N{N}_sigma{sigma}.npz")
        perr_dsts = data_dsts["perr"]
        beta = data_dsts["beta_grid"]
        mus_grid = data_dsts["mus_grid"]

        # ---------- Construct difference surface ----------
        perr_surface_dts = np.repeat(perr_dts[:, None], len(beta), axis=1)
        difference = perr_surface_dts - perr_dsts

        beta_th = np.full_like(mus_grid, np.nan, dtype=float)
        mu_min = 1/(1+2*N)

        # ---------- Find beta threshold ----------
        for i, mu in enumerate(mus_grid):
            if mu>=mu_min:
                diff_mu = difference[i, :]

                # Remove NaNs for this μ only
                mask = np.isfinite(diff_mu)

                if np.sum(mask) < 2:
                    continue

                diff_valid = diff_mu[mask]
                beta_valid = beta[mask]

                # Exact zero
                zero_idx = np.where(diff_valid == 0)[0]
                if len(zero_idx) > 0:
                    beta_th[i] = beta_valid[zero_idx[0]]
                    continue

                # Find sign changes
                sign_change = np.where(np.diff(np.sign(diff_valid)) != 0)[0]
                
                if len(sign_change) == 0:
                    continue

                # Choose which crossing you want
                j = sign_change[1] if len(sign_change) > 1 else sign_change[0]

                b1, b2 = beta_valid[j], beta_valid[j + 1]
                d1, d2 = diff_valid[j], diff_valid[j + 1]

                # Linear interpolation
                beta_th[i] = b1 + (0 - d1) * (b2 - b1) / (d2 - d1)

        beta_th_dict[f"sigma_{sigma}"] = beta_th

    return beta_th_dict


def plot_squeezing(N:float, sigmas:list, colors_opt:list, colors_th:list, opt:bool = False, th:bool = True) -> dict:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for different sigmas.
    '''

    n_gh = 100
    gauss = hermgauss(n_gh)
    mu_min = 1/(1 + 2*N)
    if opt or th:
        plt.figure(figsize=(15,6), dpi=300)
    #---------- FIND THRESHOLD ----------
    
    beta_opt_data = beta_optimal_data(N, sigmas)
    beta_th_data = beta_threshold_data(N, sigmas) 

    for i,sigma in enumerate(sigmas):

        # ------------- LOAD DATA ----------------------
               
        data_dts = np.load(f"{DATAPATH_DTS}_N{N}_sigma{sigma}.npz")
        perr_dts =  data_dts["perr"]

        data_dsts = np.load(f"{DATAPATH_DSTS}_N{N}_sigma{sigma}.npz")
        perr_dsts =  data_dsts["perr"]
        beta = data_dsts["beta_grid"]
        mus_grid = data_dsts["mus_grid"]
    
        mus_surface, beta_surface = np.meshgrid(mus_grid, beta, indexing="ij")
        perr_surface_dts = np.zeros_like(mus_surface)

        for k in range(len(mus_grid)):
            for l in range(len(beta)):

                perr_surface_dts [k, l] = perr_dts[k]

        #-------------------------  THRESHOLD  -------------------------
        # Find theoretical values for β_th
        beta_th_theory = np.full((len(mus_grid)), np.nan)
        for idx_th, m in enumerate(mus_grid):
            if m>=mu_min:
                beta_th_theory[idx_th]=(beta_threshold_theory(N, m, sigma, gauss))
        beta_th_theory= np.array(beta_th_theory)
        
        #-------------------------  R^2  THRESHOLD -------------------------

        mask_NANS = np.isfinite(beta_th_theory) & np.isfinite(beta_th_data[f'sigma_{sigma}'])

        th_theory = beta_th_theory[mask_NANS]
        th_data = beta_th_data[f'sigma_{sigma}'][mask_NANS]
    
        ss_res_th = np.sum((th_data - th_theory)**2)
        ss_tot_th = np.sum((th_data - np.mean(th_data))**2)
        R2_th = 1 - ss_res_th/ss_tot_th
        
        # ------------- OPTIMAL ----------------------

        # Find theoretical values for β_opt
        beta_opt_theory = np.full((len(mus_grid)), np.nan)
        for idx_opt, m in enumerate(mus_grid):
            if m>=mu_min:
                beta_opt_theory[idx_opt] = beta_optimal_theory(N, m, sigma, gauss)
        
        #-------------------------  R^2 OPTIMAL  -------------------------

        mask_NANS = np.isfinite(beta_opt_theory) & np.isfinite(beta_opt_data[f'sigma_{sigma}'])

        opt_theory = beta_opt_theory[mask_NANS]
        opt_data = beta_opt_data[f'sigma_{sigma}'][mask_NANS]

        ss_res_opt = np.sum((opt_data[1:] - opt_theory[1:])**2)
        ss_tot_opt = np.sum((opt_data[1:] - np.mean(opt_data[1:]))**2)
        R2_opt = 1 - ss_res_opt/ss_tot_opt

        if opt or th:
            plt.xlabel(r'$\mu$ (Purity of seed state)')
            plt.ylabel(r'$\beta$ (Squeezing Fraction)')
            
        if th:
            plt.scatter(mus_grid, beta_th_data[f'sigma_{sigma}'], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, 
                        label = fr'$\beta_{{\rm th}}$: σ = {sigma} : R2 = {R2_th:.2f}')
            plt.fill_between(mus_grid, beta_th_theory, 0, alpha=0.8, zorder=0, color=colors_th[i])
            plt.plot(mus_grid, beta_th_theory, color='k', linewidth = 3)

        if opt:
            plt.scatter(mus_grid[1:], beta_opt_data[f'sigma_{sigma}'][1:], color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, 
                        label = fr'$\beta_{{\rm opt}}$: σ = {sigma} : R2 = {R2_opt:.2f}')
            plt.plot(mus_grid[1:], beta_opt_theory[1:], color='k', linewidth = 3)
        
        if th or opt:
            plt.legend()
            plt.tight_layout()

    if (not th) and (not opt):
        plt.clf()

#======================================================================================
#                           SIGMA THRESHOLD 
#======================================================================================

def sigma_threshold_theory(N:float, mu_grid:float) -> np.array:

    gauss = hermgauss(100)

    def F(sigma):

        beta_opt = beta_optimal_theory(N, mu, sigma, gauss)
        p_err_dsts = theory_point_dsts(N, beta_opt, mu, sigma, gauss)
        p_err_dts = theory_point_dts(N, mu, sigma, gauss)
        return p_err_dsts - p_err_dts
     
    min_mu = 1/(1 + 2*N)
    sigma_low, sigma_high = 0, 1
    sigma_th = np.full(len(mu_grid), np.nan)

    for i, mu in enumerate(mu_grid):

        if mu > min_mu:

            try:
                sigma_th[i] = brentq(F, sigma_low, sigma_high)
            except:
                 sigma_th[i] = 1

    return sigma_th


def sigma_threshold_data(N:float, sigmas:np.array, mus:np.array) -> np.array:

    mu_min = 1/(1 + 2*N)

    betas = beta_optimal_data(N, sigmas)
    sigma_th_data = np.ones((len(mus)))

    for sigma_key in betas.keys():
        if 0 in betas[sigma_key] :
            mu_indices = np.where(betas[sigma_key] == 0)[0]
            sigma_th = float(sigma_key.split('_')[1])

            for l in mu_indices:
                if mus[l] > mu_min:
                    if sigma_th_data[l] == 1:
                        sigma_th_data[l] = sigma_th

    for l,m in enumerate(mus): 
        if mus[l] < mu_min:
            sigma_th_data[l] = np.nan
    
    return sigma_th_data

def plot_sigma_threshold(N_vals:np.array, sigmas, mus_grid, colors_th, colors_points):

    sigma_thresholds = np.zeros((len(N_vals), len(mus_grid)))
    sigma_thresholds_data = np.zeros((len(N_vals), len(mus_grid)))

    for i, N in enumerate(N_vals):
        sigma_thresholds[i] = sigma_threshold_theory(N, mus_grid)
        sigma_thresholds_data[i] = sigma_threshold_data(N, sigmas, mus_grid)

    plt.figure(figsize=(3, 3), dpi=200)
    
    for i, N in enumerate(N_vals):
        plt.plot(mus_grid, sigma_thresholds[i], color=colors_th[i], linewidth=3, label=f'N={N}')
        plt.fill_between(mus_grid, sigma_thresholds[i], 0, color=colors_th[i], alpha=0.5)
        plt.axvline(x=1/(1+2*N), linestyle='--', color=colors_th[i])

        plt.plot(mus_grid, sigma_thresholds_data[i], '.', color = colors_points[i], alpha = 0.6)

    plt.ylim(-0.01, 1.01)
    plt.xlim(0.05, 1.01)
    
    plt.xlabel('μ')
    plt.ylabel(r'$\sigma_{th}$')
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.show()

#======================================================================================
#                                   HELSTROM
#======================================================================================

def helstrom_bound(N:float, mus_grid:np.array, sigma:float, fock_cutoff:int, dsts = True):

    mu_min = 1/(1+2*N)
    n_gh = 100
    gauss = hermgauss(n_gh)
    x, w = gauss

    beta_opt_array = np.full_like(mus_grid, np.nan, dtype=float)
    for i,mu in enumerate(mus_grid):
        if mu>mu_min:
            beta_opt_array[i] = beta_optimal_theory(N, mu, sigma, gauss)

    if not dsts:
        beta_opt_array = np.zeros((len(mus_grid)))

    r_opt = np.arcsinh(np.sqrt(N*beta_opt_array))
    alpha = np.full_like(mus_grid, np.nan, dtype=float)
    valid = mus_grid > mu_min
    alpha[valid] = np.sqrt(N*(1-beta_opt_array[valid])+(mus_grid[valid]-1)*(1+2*N*beta_opt_array[valid])/(2*mus_grid[valid]))
    
    def state(alpha, r, mu, phi):
        Nth = (1 - mu)/(2*mu)
        prog = sf.Program(1)
        with prog.context as q:
            Thermal(Nth) | q
            Sgate(r) | q
            Dgate(alpha) | q
            Rgate(phi) | q
        #run the engine and get the state
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(prog)
        return result.state.dm()

    p_helstrom = np.full_like(mus_grid, np.nan, dtype=float)
    phis = np.sqrt(2)*sigma*x

    for j,mu in enumerate(mus_grid):
        if mu>mu_min:
            rho_1 = 0
            rho_2 = 0

            for i, phi in enumerate(phis):
                rho_1 +=  w[i]*state(alpha[j], r_opt[j], mu, phi)/np.sqrt(np.pi)
                rho_2 +=  w[i]*state(-1*alpha[j], r_opt[j], mu, phi)/np.sqrt(np.pi)

            Delta = rho_1 - rho_2
            eigenvals = np.linalg.eigvalsh(Delta)
            trace_norm = np.sum(np.abs(eigenvals))

            p_helstrom[j] = 0.5*(1 - 0.5*trace_norm)
    
    return p_helstrom

def p_err_calculation(N, mus_grid, sigma, fock_cutoff):
    p_helstrom_dsts = helstrom_bound(N, mus_grid, sigma, fock_cutoff, dsts=True)
    p_helstrom_dts = helstrom_bound(N, mus_grid, sigma, fock_cutoff, dsts=False)

    n_gh = 100
    gauss = hermgauss(n_gh)
    p_dsts_hd = np.full_like(mus_grid, None, dtype=float)
    p_dts_hd = np.full_like(mus_grid, None, dtype=float)
    mu_min = 1/(1+2*N)

    for i,mu in enumerate(mus_grid):
        if mu>mu_min:
            beta_opt_theory = beta_optimal_theory(N, mu, sigma, gauss)
            p_dsts_hd[i] = theory_point_dsts(N, beta_opt_theory, mu, sigma, gauss)
            p_dts_hd[i] = theory_point_dts(N, mu, sigma, gauss)
    return (p_helstrom_dsts, p_helstrom_dts), (p_dsts_hd, p_dts_hd)


def plot_helstrom_vs_homodyne(p_helstrom, p_hd, N, sigma, mus_grid):

    p_helstrom_dsts, p_helstrom_dts = p_helstrom
    p_dsts_hd, p_dts_hd = p_hd

    # Find sigma threshold
    sigma_th = sigma_threshold_theory(N, mus_grid)
    th=1e-3
    idx = np.where((sigma_th > sigma - th) & (sigma_th < sigma + th))[0]
    try:
        mu_th = (mus_grid[idx[0]]+mus_grid[idx[-1]])/2
    except:
        mu_th = 1

    # Find regions where squeezing is beneficial
    difference_hd = p_dsts_hd - p_dts_hd
  
    idx_pos = np.where(difference_hd>0)[0]
    idx_neg = np.where(difference_hd<0)[0]
    if len(idx_pos) == 0:
        idx_pos = idx_neg

    #-------------------------------- PLOT --------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(rf'$N={N}$, $\sigma={sigma}$', fontsize=16)

    ax[0].set_title('DSTS')
    ax[0].plot(mus_grid, p_helstrom_dsts, linestyle='--', color='k', label='Helstrom')
    ax[0].plot(mus_grid, p_dsts_hd, linestyle='-', color='b', label='Homodyne')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$P_{err}$')

    ax[1].set_title(rf'DTS')
    ax[1].plot(mus_grid, p_helstrom_dts, linestyle='--', color='k', label='Helstrom')
    ax[1].plot(mus_grid, p_dts_hd, linestyle='-', color='b', label='Homodyne')
    ax[1].set_yscale('log')

    for axis in ax:
        axis.set_ylim(np.nanmin(p_helstrom_dsts)/2, 2*np.nanmax(p_helstrom_dsts))
        axis.legend()
        axis.set_xlabel(r'$\mu$')
    plt.show()

    #-------------------------------- PLOT --------------------------------
    
    fig, ax = plt.subplots(2, 1, figsize=(10,7), dpi=100)

    fig.suptitle(rf'$N={N}$, $\sigma={sigma}$', fontsize=16)

    ax[0].plot(mus_grid, p_helstrom_dsts-p_helstrom_dts, linestyle='--', color='k', label='Helstrom')
    ax[0].plot(mus_grid, p_dsts_hd-p_dts_hd, linestyle='-', color= 'b', label='Homodyne')
    ax[0].axvline(x=mu_th, color = "#AD0B90", linestyle = '--', label = r'$\sigma_{th}$ is reached', alpha=0.30)
    ax[0].axvspan(mus_grid[idx_neg[0]], mus_grid[idx_neg[-1]], color='blue', alpha=0.10, label = 'Squeezing beneficial')
    ax[0].axvspan(mus_grid[idx_neg[-1]], mus_grid[idx_pos[-1]], color= "#AD0B90", alpha=0.10, label = 'Squeezing not beneficial')
    ax[0].set_ylabel(r'$P^{(DSTS)}_{err}-P^{(DTS)}_{err}$')
    ax[0].set_xlabel(r'$\mu$')
    ax[0].legend()
    ax[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)

    ax[1].plot(mus_grid, sigma_th, color='k', alpha=0.5 )
    ax[1].fill_between(mus_grid[mus_grid <= mu_th], sigma_th[mus_grid <= mu_th], 0, color='blue', alpha=0.10)
    ax[1].fill_between(mus_grid[mus_grid >= mu_th], sigma_th[mus_grid >= mu_th], 0, color="#AD0B90", alpha=0.10)
    ax[1].set_xlabel('μ')
    ax[1].set_ylabel(r'$\sigma_{th}$')
    ax[1].axhline(y=sigma, color ='gray', linestyle = '--', alpha=0.5)
    ax[1].axvline(x=mu_th, color = "#AD0B90", linestyle = '--', alpha=0.50)
    plt.tight_layout()
    plt.show()

    #-------------------------------- PLOT --------------------------------

    fig, ax = plt.subplots(1, 1, figsize=(10,7), dpi=100)

    fig.suptitle(rf'$N={N}$, $\sigma={sigma}$', fontsize=16)

    ax.plot(mus_grid, abs(p_dsts_hd-p_helstrom_dsts), linestyle='-', color='k', label='DSTS')
    ax.plot(mus_grid, abs(p_dts_hd-p_helstrom_dts), linestyle='-', color= 'b', label='DTS')
    ax.axvline(x=mu_th, color = "#AD0B90", linestyle = '--', label = r'$\sigma_{th}$ is reached', alpha=0.30)
    ax.axvspan(mus_grid[idx_neg[0]], mus_grid[idx_neg[-1]], color='blue', alpha=0.10, label = 'Squeezing beneficial')
    ax.axvspan(mus_grid[idx_neg[-1]], mus_grid[idx_pos[-1]], color= "#AD0B90", alpha=0.10, label = 'Squeezing not beneficial')
    ax.set_ylabel(r'|$P^{(min)}_{err}-P^{(HD)}_{err}$|')
    ax.set_xlabel(r'$\mu$')
    ax.legend()
    plt.tight_layout()
    plt.show()