import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
import math
from scipy.special import erfc
from scipy.optimize import brentq, minimize_scalar
import plotly.graph_objects as go
from helper_functions.state_measurement import *
import random

# CONTENTS:
#==================================
# ERROR PROBABILITY
# BETA THRESHOLD & OPTIMAL SQUEEZING
# HELSTROM

global DATAPATH_CS, DATAPATH_DSS
DATAPATH_CS = 'data/CS/perr_cs_a101_S1000000000'
DATAPATH_DSS = 'data/DSS/perr_dss_N101_b101_S1000000000'

#======================================================================================
#                           ERROR PROBABILITY
#======================================================================================

def perr_cs(alpha_grid:np.array, sigma:float, num_samples:int, strawberry:bool=False) -> np.array:
    '''
    Experimental calculation of the homodyne error probability for coherent states
    '''
    #Calculation of error probability
    #============================================
    p_err = np.zeros((len(alpha_grid)))

    if not strawberry:
        
        gauss = hermgauss(100)

        for i, N in enumerate(alpha_grid**2):


            p = theory_point_cs(N, sigma, gauss)
            p_err[i] = np.random.binomial(num_samples, p) / num_samples

    #============================================
    else:

        eng = sf.Engine("gaussian")

        for i in range(len(alpha_grid)):

            # Choose phase from Gaussian distribution
            #============================================
            phis = np.random.normal(0, sigma, size = num_samples) 
            wrong_sign_counter = 0

            for phi in phis:

                coherent_sign = np.random.choice([1, -1])
                prog = sf.Program(1)

                with prog.context as q:
                    Dgate(coherent_sign*alpha_grid[i]) | q
                    Rgate(phi) | q
                    MeasureHomodyne(0) | q

                result = eng.run(prog)
                eng.reset()
                result_sign = np.sign(result.samples[0][0])

                if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                    wrong_sign_counter+= 1

            p_err[i] = wrong_sign_counter/num_samples
        
    return p_err


def perr_dss(N_grid:np.array, beta_grid:np.array, sigma:float, num_samples:int, strawberry:bool=False) -> np.array:
    '''
    Experimental calculation of the homodyne error probability for displaced squeezed states
    '''
    #Calculation of error probability
    #============================================
    p_err = np.zeros((len(N_grid), len(beta_grid)))

    if not strawberry:

        gauss = hermgauss(100)

        for i, N in enumerate(N_grid):

            for k, beta in enumerate(beta_grid):
                p = theory_point_dss(N, beta, sigma, gauss)
                p_err[i, k] = np.random.binomial(num_samples, p) / num_samples
    #============================================
    else:

        eng = sf.Engine("gaussian")

        for i, N in enumerate(N_grid):

            alphas = np.sqrt(N*(1-beta_grid)) 


            for k, beta in enumerate(beta_grid):

                wrong_sign_counter = 0
                r_s = math.asinh(np.sqrt(N*beta))

                # Choose phase from Gaussian distribution
                #============================================
                phis = np.random.normal(0, sigma, size = num_samples)

                for phi in phis:

                    coherent_sign = np.random.choice([1, -1])
                    prog = sf.Program(1)

                    with prog.context as q:
                        Sgate(r_s, 0) | q[0] 
                        Dgate(coherent_sign*alphas[k]) | q
                        Rgate(phi) | q
                        MeasureHomodyne(0) | q

                    result = eng.run(prog)
                    eng.reset()
                    result_sign = np.sign(result.samples[0][0])

                    if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                        wrong_sign_counter+= 1

                p_err[i][k] = wrong_sign_counter/num_samples

    return p_err



def theory_point_cs(N:float, sigma:float, gauss:tuple) -> float:
        '''
        Numerical calculation of the theoretical homodyne error probability for coherent states,
        works for a single (N, β) points.
        '''
        x_gh, w_gh = gauss
        a = np.sqrt(N)
        r = 0
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = np.sqrt(2)*a*np.cos(phi)/np.sqrt(V)
        integral = 0.5*np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return integral


def theory_point_dss(N:float, beta:float, sigma:float, gauss:tuple) -> float:
        '''
        Numerical calculation of the theoretical homodyne error probability for displaced squeezed states,
        works for a single (N, β) points.
        '''
        x_gh, w_gh = gauss

        a = np.sqrt(N*(1-beta))
        r = np.arcsinh(np.sqrt(N * beta))
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r) * np.cos(phi)**2 + np.exp(2*r) * np.sin(phi)**2)

        arg = np.sqrt(2) * a * np.cos(phi)/np.sqrt(V)
        integral = 0.5*np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return integral


def plot_homodyne_perr(sigmas:list, colors_light:list, colors_dark:list, cs:str|bool = False, dss:str|bool = False) -> None:
    '''
    Plots the homodyne error probability for CS and DSS.
    CS and DSS can only take the values 'data', 'theory', 'all', or False
    '''

    valid_args = {'data', 'theory', 'all', False}

    if cs not in valid_args or dss not in valid_args:
         raise KeyError("CS and DSS can only take the values 'data', 'theory', 'all', or False")

    n_gh = 100
    gauss = hermgauss(n_gh)

    fig = go.Figure()
    
    #============================  CS  ============================
    
    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_cs = np.load(f"{DATAPATH_CS}_sigma{sigma}.npz")

        alpha_cs = data_cs["alpha_grid"]
        perr_cs =  data_cs["p_err_cs"]
        sigma_cs = data_cs["sigma"]

        N_cs = alpha_cs**2
        beta_cs = np.linspace(0, 1, len(N_cs))
        
        N_surface_cs, beta_surface_cs = np.meshgrid(N_cs, beta_cs, indexing="ij")
        z_surface_cs = np.zeros_like(N_surface_cs)
        perr_surface_cs = np.zeros_like(N_surface_cs)

        #-------------------------  Theoretical curve  -------------------------
        for k in range(len(N_cs)):
            for l in range(len(beta_cs)):

                perr_surface_cs [k, l] = perr_cs[k]
                z_surface_cs[k, l] = theory_point_cs(N_cs[k],  sigma_cs, gauss)
                
        #-------------------------  R^2  -------------------------
        ss_res_cs = np.sum((perr_cs - z_surface_cs[:,0])**2)
        ss_tot_cs = np.sum((perr_cs - np.mean(perr_cs))**2)
        R2_cs = 1 - ss_res_cs/ss_tot_cs

        #-------------------------  Plot  -------------------------
        if cs in ['theory', 'all']:
        
            fig.add_trace(go.Surface(x=N_surface_cs, y=beta_surface_cs, z=z_surface_cs, surfacecolor=np.zeros_like(z_surface_cs), 
                                colorscale=[[0.0, colors_light[i]], [1.0, colors_light[i]]], showscale=False))
            
        if cs in ['data', 'all']:
            
            fig.add_trace(go.Scatter3d(x=N_surface_cs.ravel(), y=beta_surface_cs.ravel(), z=perr_surface_cs.ravel(), mode="markers", 
                                marker=dict(size=3, color=colors_dark[i]), name=f"CS| σ={sigma_cs}: R2 = {R2_cs:0.3f}"))


    #============================  DSS  ============================

    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_dss = np.load(f"{DATAPATH_DSS}_sigma{sigma}.npz")
        N_dss =  data_dss["N"]
        beta_dss =  data_dss["beta"]
        perr_dss = data_dss["p_err_dss"]
        sigma_dss = data_dss["sigma"]

        N_surface_dss, beta_surface_dss = np.meshgrid(N_dss, beta_dss, indexing="ij")
        z_surface_dss = np.zeros_like(N_surface_dss)

        #-------------------------  Theoretical curve  -------------------------
        for k in range(len(N_dss)):
            for l in range(len(beta_dss)):

                z_surface_dss[k, l] = theory_point_dss(N_dss[k], beta_dss[l], sigma_dss, gauss)

        #-------------------------  R^2  -------------------------
        ss_res_dss = np.sum((perr_dss - z_surface_dss)**2)
        ss_tot_dss = np.sum((perr_dss - np.mean(perr_dss))**2)
        R2_dss = 1 - ss_res_dss/ss_tot_dss

        #-------------------------  Plot  -------------------------
        if dss in ['theory', 'all']:
                fig.add_trace(go.Surface( x=N_surface_dss, y=beta_surface_dss, z=z_surface_dss, surfacecolor=np.zeros_like(z_surface_dss), 
                                colorscale=[[0.0, colors_dark[i]], [1.0, colors_dark[i]]], showscale=False))
        
        if dss in ['data', 'all']:
                fig.add_trace(go.Scatter3d(x=N_surface_dss.ravel(), y=beta_surface_dss.ravel(), z=perr_dss.ravel(),
                mode="markers", marker=dict(size=3, color=colors_light[i]), name=f"DSS| σ={sigma_dss}: R2 = {R2_dss:0.3f}"))

        fig.update_layout(scene=dict(xaxis_title="N", yaxis_title=r"β", zaxis = dict(title="P_err", type="log"), 
                            aspectmode ="cube"), width=900, height=750)
        
    if cs or dss:
        fig.show()

#======================================================================================
#                        BETA THRESHOLD & OPTIMAL SQUEEZING
#======================================================================================

def beta_threshold_theory(N:float, sigma:float, gauss:tuple) -> float:
    '''
    Solves the equation F(β_th) = theory_point_dss - theory_point_cs = 0 for a fixed value of N and σ,
    given the fitted parameters of the models. 
    
    The search for the root is conducted in the interval (β_min, 1) to ensure that the first root for β=0 is not considered,
    where dF(β_min)/dβ = 0.
    '''
    point_cs = theory_point_cs(N, sigma, gauss)
    
    def F(beta):
        
        point_dss = theory_point_dss(N, beta, sigma, gauss) 

        return point_dss - point_cs
    
    # sample to locate minimum
    #=========================
    beta_grid = np.linspace(0, 1, 500)
    F_b = np.array([F(b) for b in beta_grid])
    beta_min = beta_grid[np.argmin(F_b)]
    beta_max = 1

    try:
        beta_upper = brentq(F, beta_min, beta_max)
        return beta_upper
    
    except:
        return 0.0


def beta_optimal_theory(N:float, sigma:float, gauss:tuple) -> float:
    '''
    Utilized an optimization method to find the value of β in [0,1] which minimizes the homodyne error probability;
    thus, the optimal value of β.
    '''
    def objective(beta:float):
        '''
        Defined the objective function for the minimization, which is the homodyne error probability
        '''
        return theory_point_dss(N, beta, sigma, gauss)

    res = minimize_scalar(objective, bounds=(0,1), method="bounded")
    return res.x


def beta_optimal_data(sigmas:list) -> dict:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for different sigmas.
    '''
    #---------- FIND THRESHOLD ----------
    beta_opt_dict = {}

    for i,sigma in enumerate(sigmas):

        # ------------- LOAD DATA ----------------------
               
        data_cs = np.load(f"{DATAPATH_CS}_sigma{sigma}.npz")
        perr_cs =  data_cs["p_err_cs"]

        data_dss = np.load(f"{DATAPATH_DSS}_sigma{sigma}.npz")
        perr_dss =  data_dss["p_err_dss"]
        beta = data_dss["beta"]
        N_grid = data_dss["N"]
    
        mus_surface, beta_surface = np.meshgrid(N_grid, beta, indexing="ij")
        perr_surface_cs = np.zeros_like(mus_surface)

        for k in range(len(N_grid)):
            for l in range(len(beta)):

                perr_surface_cs [k, l] = perr_cs[k]
        
        # ------------- OPTIMAL ----------------------

        # Minima along beta for each N
        valid_rows = ~np.all(np.isnan(perr_dss), axis=1)
        beta_opt = np.full(len(N_grid), np.nan)
        idx = np.nanargmin(perr_dss[valid_rows], axis=1)
        beta_opt[valid_rows] = beta[idx]
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt
        
    return beta_opt_dict

def beta_threshold_data(sigmas: list) -> dict:
    """
    Finds the threshold β_th by locating the zero crossing of
    perr_dts - perr_dsts along the beta axis for each μ.

    NaN values are ignored independently for each μ.
    """

    beta_th_dict = {}

    for sigma in sigmas:

        # ---------- Load data ----------
        data_cs = np.load(f"{DATAPATH_CS}_sigma{sigma}.npz")
        perr_cs =  data_cs["p_err_cs"]

        data_dss = np.load(f"{DATAPATH_DSS}_sigma{sigma}.npz")
        perr_dss =  data_dss["p_err_dss"]
        beta = data_dss["beta"]
        N_grid = data_dss["N"]

        # ---------- Construct difference surface ----------
        perr_surface_dts = np.repeat(perr_cs[:, None], len(beta), axis=1)
        difference = perr_surface_dts - perr_dss

        beta_th = np.full_like(N_grid, np.nan, dtype=float)

        # ---------- Find beta threshold ----------
        for i, n in enumerate(N_grid):
                diff_N = difference[i, :]

                # Remove NaNs for this μ only
                mask = np.isfinite(diff_N)

                if np.sum(mask) < 2:
                    continue

                diff_valid = diff_N[mask]
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


def plot_squeezing(sigmas:list, colors_opt:list, colors_th:list, opt:bool = False, th:bool = True) -> dict:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for different sigmas.
    '''

    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(15,6), dpi=300)

    #---------- FIND THRESHOLD ----------
    beta_opt_dict = beta_optimal_data(sigmas)
    beta_th_dict = beta_threshold_data(sigmas)

    for i,sigma in enumerate(sigmas):

        # ------------- LOAD DATA ----------------------
        data_cs = np.load(f"{DATAPATH_CS}_sigma{sigma}.npz")
        perr_cs =  data_cs["p_err_cs"]

        data_dss = np.load(f"{DATAPATH_DSS}_sigma{sigma}.npz")
        N =  data_dss["N"]
        beta =  data_dss["beta"]
        perr_dss = data_dss["p_err_dss"]

        N_surface, beta_surface = np.meshgrid(N, beta, indexing="ij")
        perr_surface_cs = np.zeros_like(N_surface)

        for k in range(len(N)):
            for l in range(len(beta)):

                perr_surface_cs [k, l] = perr_cs[k]


        #-------------------------  THRESHOLD  -------------------------

        # Find theoretical values for β_th
        beta_th_theory = []
        for n in N:
            beta_th_theory.append(beta_threshold_theory(n, sigma, gauss))
        beta_th_theory= np.array(beta_th_theory)

        #-------------------------  R^2  THRESHOLD -------------------------

        beta_th_data = beta_th_dict[f'sigma_{sigma}']
        mask_NANS = np.isfinite(beta_th_theory) & np.isfinite(beta_th_data)

        th_theory = beta_th_theory[mask_NANS]
        th_data = beta_th_dict[f'sigma_{sigma}'][mask_NANS]
    
        ss_res_th = np.sum((th_data[1:] - th_theory[1:])**2)
        ss_tot_th = np.sum((th_data[1:] - np.mean(th_data[1:]))**2)
        R2_th = 1 - ss_res_th/ss_tot_th

        # ------------- OPTIMAL ----------------------
        beta_opt_theory = np.zeros_like(N)

        for idx, n in enumerate(N):
            beta_opt_theory[idx] = beta_optimal_theory(n, sigma, gauss)

        #-------------------------  R^2 OPTIMAL  -------------------------
        mask_NANS = np.isfinite(beta_opt_theory) & np.isfinite(beta_opt_dict[f'sigma_{sigma}'])

        opt_theory = beta_opt_theory[mask_NANS]
        opt_data = beta_opt_dict[f'sigma_{sigma}'][mask_NANS]
        beta_opt = beta_opt_dict[f'sigma_{sigma}']
        ss_res_opt = np.sum((opt_data[1:] - opt_theory[1:])**2)
        ss_tot_opt = np.sum((opt_data[1:] - np.mean(opt_data[1:]))**2)
        R2_opt = 1 - ss_res_opt/ss_tot_opt


        if th:
            
            plt.scatter(N[1:], beta_th_data[1:], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, 
                        label = fr'$\beta_{{\rm th}}$: σ = {sigma}, $R^2$ = {R2_th:0.3f}')
            plt.fill_between(N, beta_th_theory, 0, alpha=0.8, zorder=0, color=colors_th[i])
            plt.plot(N, beta_th_theory, color='k', linewidth = 3)

        if opt:
            
            plt.scatter(N[1:], beta_opt[1:], color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, 
                        label = fr'$\beta_{{\rm opt}}$: σ = {sigma}, $R^2$ = {R2_opt:0.3f}')
            plt.plot(N[1:], beta_opt_theory[1:], color='k', linewidth = 3)

        if th or opt:
            plt.legend()
            plt.tight_layout()
            plt.xlabel(r'$N$ (Average number of photons)')
            plt.ylabel(r'$\beta$ (Squeezing Fraction)')
            plt.ylim(0,1)

        if (not th) and (not opt):
            plt.clf()
        


def optimal_squeezing_noisefree(opt:bool = True, th:bool = True) -> None:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for the noisefree case.
    '''

    plt.figure(figsize=(15,6), dpi=300)

    #---------- FIND THRESHOLD ----------

    data_cs = np.load(f"{DATAPATH_CS}_sigma0.0.npz")
    perr_cs =  data_cs["p_err_cs"]

    data_dss = np.load(f"{DATAPATH_DSS}_sigma0.0.npz")
    N =  data_dss["N"]
    beta =  data_dss["beta"]

    beta_cs = np.linspace(0, 1, len(N))
    N_surface_cs, beta_surface_cs = np.meshgrid(N, beta_cs, indexing="ij")
    perr_surface_cs = np.zeros_like(N_surface_cs)

    for k in range(len(N)):
        for l in range(len(beta_cs)):

            perr_surface_cs [k, l] = perr_cs[k]

    # ------------- THRESHOLD ----------------------

    beta_th_data = beta_threshold_data([0.0])['sigma_0.0']
    beta_th_theory = 4*N/(4*N + 1)

    #-------------------------  R^2 THRESHOLD  -------------------------

    ss_res_th = np.sum((beta_th_data - beta_th_theory)**2)
    ss_tot_th = np.sum((beta_th_data - np.mean(beta_th_data))**2)
    R2_th = 1 - ss_res_th/ss_tot_th

    # ------------- OPTIMAL ----------------------
    
    beta_opt_data = beta_optimal_data([0.0])['sigma_0.0']
    beta_opt_line =  N/(2*N + 1)

    #-------------------------  R^2 OPTIMAL  -------------------------
    ss_res_opt = np.sum((beta_opt_data[1:] - beta_opt_line[1:])**2)
    ss_tot_opt = np.sum((beta_opt_data[1:] - np.mean(beta_opt_data[1:]))**2)
    R2_opt = 1 - ss_res_opt/ss_tot_opt

    if th:
        plt.scatter(N, beta_th_data, s=30, edgecolors='k', color='white', marker='D', zorder=10, 
                    label = fr'$\beta_{{\rm th}}$: $R^2$ = {R2_th:0.3f}')
        plt.fill_between(N, beta_th_theory, 0, alpha=0.8, zorder=0, color='blue')
        plt.plot(N, beta_th_theory, color='k', linewidth = 3)

    if opt:
        plt.scatter(N[1:], beta_opt_data[1:], color='red', edgecolors='k', s=50, marker='o', zorder=10, 
                    label = fr'$\beta_{{\rm opt}}$: $R^2$ = {R2_opt:0.3f}')
        plt.plot(N[1:], beta_opt_line[1:], color='k', linewidth = 3)

        if not th:
             plt.legend()

    if th:
        plt.legend()
        plt.xlabel(r'$N$ (Average number of photons)')
        plt.ylabel(r'$\beta$ (Squeezing Fraction)')
        plt.tight_layout()

    if (not th) and (not opt):
        plt.clf()
        


def beta_threshold_vs_sigma(N_values: np.ndarray, sigmas: np.ndarray, colors_1, colors_2):

    n_gh = 100
    gauss = hermgauss(n_gh)

    # Compute threshold data once
    beta_th_data_dict = beta_threshold_data(sigmas)

    # Load N grid once
    data = np.load(f"{DATAPATH_DSS}_sigma{sigmas[0]}.npz")
    N_grid = data["N"]

    plt.figure(figsize=(6, 5), dpi=300)

    for i, N in enumerate(N_values):

        # Find corresponding index in the data
        idx = np.argmin(np.abs(N_grid - N))

        beta_th_theory = []
        beta_th_data = []

        for sigma in sigmas:

            # Theory
            beta_th = beta_threshold_theory(N, sigma, gauss)
            if np.isnan(beta_th):
                beta_th = 0.0
            beta_th_theory.append(beta_th)

            # Data
            beta_th_data.append(beta_th_data_dict[f"sigma_{sigma}"][idx])

        beta_th_theory = np.array(beta_th_theory)
        beta_th_data = np.array(beta_th_data)

        # R²
        mask = np.isfinite(beta_th_data)
        ss_res = np.sum((beta_th_data[mask] - beta_th_theory[mask])**2)
        ss_tot = np.sum((beta_th_data[mask] - np.mean(beta_th_data[mask]))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        # Plot
        plt.fill_between(sigmas, beta_th_theory, 0, color =colors_1[i], alpha=0.30)
        plt.plot(sigmas, beta_th_theory, color =colors_1[i], linewidth=2)
        plt.scatter(sigmas, beta_th_data, color =colors_2[i], marker='D', s=20, edgecolors='k', zorder=10, label=fr"$N={N}$, $R^2={R2:.3f}$")

    plt.ylim(-0.01, 1.01)
    plt.xlim(sigmas[0], sigmas[-1])
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$\beta_{\rm th}$")
    plt.legend()
    plt.tight_layout()
    plt.show()

#======================================================================================
#                                   HELSTROM
#======================================================================================


def helstrom_bound(N:float, sigmas:np.array, fock_cutoff:int, cs = False):

    n_gh = 100
    gauss = hermgauss(n_gh)
    x, w = gauss

    if cs:
        beta_opt_array = np.zeros((len(sigmas)))
    else:
        beta_opt_array = np.zeros((len(sigmas)))
        for i,sigma in enumerate(sigmas):
            beta_opt_array[i] = beta_optimal_theory(N, sigma, gauss)


    alpha = np.sqrt(N*(1-beta_opt_array)) 
    r_opt = np.arcsinh(np.sqrt(N*beta_opt_array))

    def state(alpha, r, phi):

        prog = sf.Program(1)
        with prog.context as q:
            Vac | q[0]
            Sgate(r) | q[0] 
            Dgate(alpha) | q[0]
            Rgate(phi) | q
        #run the engine and get the state
        eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
        result = eng.run(prog)
        return result.state.dm()
    

    p_helstrom = np.zeros((len(sigmas)))

    for j,sigma in enumerate(sigmas):

        phis = np.sqrt(2)*sigma*x
        rho_1 = 0
        rho_2 = 0

        for i,phi in enumerate(phis):
            rho_1 +=  w[i]*state(alpha[j], r_opt[j], phis[i])/np.sqrt(np.pi)
            rho_2 +=  w[i]*state(-1*alpha[j], r_opt[j], phis[i])/np.sqrt(np.pi)

        Delta = rho_1 - rho_2
        eigenvals = np.linalg.eigvalsh(Delta)
        trace_norm = np.sum(np.abs(eigenvals))

        p_helstrom[j] = 0.5*(1 - 0.5*trace_norm)
    
    return p_helstrom


def perr_vs_sigma(N:float, sigmas:np.array, cs:bool):

    N_grid = np.linspace(0, 2, 101)
    p = np.zeros((len(sigmas)))
    beta_opt_array = np.zeros((len(sigmas)))
    n_gh = 100
    gauss = hermgauss(n_gh)
    x, w = gauss

    if cs:
        for i, sigma in enumerate(sigmas):
        
            p[i] = theory_point_cs(N=N, sigma = sigma, gauss = (x, w))
    else:
        beta_opt_array = np.zeros((len(sigmas)))
        for i, sigma in enumerate(sigmas):
            beta_opt_array = beta_optimal_theory(N, sigma, gauss)
            p[i] = theory_point_dss(N=N, beta = beta_opt_array, sigma = sigma, gauss = (x, w))

    return p


def plot_helstrom_vs_homodyne(N:float, sigma:list, p_helstrom:tuple, p_homodyne:tuple):

    p_helstrom_cs, p_helstrom_dss = p_helstrom
    p_cs_hd, p_dss_hd = p_homodyne

    #---------------- PLOT ------------

    fig, ax = plt.subplots(1, 2, figsize=(10,5), dpi=100)
    fig.suptitle(rf'$N={N}$', fontsize=16)
    ax[0].set_title('DSS')
    ax[0].plot(sigma, p_helstrom_dss, linestyle='--', color='k', label='Helstrom')
    ax[0].plot(sigma, p_dss_hd, linestyle='-', color='b', label='Homodyne')
    ax[0].set_yscale('log')

    ax[1].set_title('CS')
    ax[1].plot(sigma, p_helstrom_cs, linestyle='--', color='k', label='Helstrom')
    ax[1].plot(sigma, p_cs_hd, linestyle='-', color='b', label='Homodyne')
    ax[1].set_yscale('log')

    for axis in ax:
        axis.set_ylim(min(p_helstrom_dss)/2, 2*max(p_helstrom_dss))
        axis.legend()
        axis.set_xlabel(r'$\sigma$')
        axis.set_ylabel(r'$P_{err}$')
    plt.tight_layout
    plt.show()

    #---------------- PLOT ------------
    # Find regions where squeezing is beneficial
    difference_hd = p_dss_hd - p_cs_hd
   
    idx_neg = np.where(difference_hd<0)[0]
    idx_pos = np.where(difference_hd>0)[0]
    if len(idx_pos) == 0:
            idx_pos = idx_neg

    fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi=100)
    fig.suptitle(rf'$N={N}$', fontsize=16)
    ax.plot(sigma, p_helstrom_dss-p_helstrom_cs, linestyle='--', color='k', label='Helstrom')
    ax.plot(sigma, p_dss_hd-p_cs_hd, linestyle='-', color='b', label='Homodyne') 
    ax.legend()
    ax.axvspan(sigma[idx_neg[0]], sigma[idx_neg[-1]], color='blue', alpha=0.10, label = 'Squeezing beneficial')
    ax.axvspan(sigma[idx_neg[-1]], sigma[idx_pos[-1]], color= "#AD0B90", alpha=0.10, label = 'Squeezing not beneficial')
    ax.axhline(y=0, color ='gray', linestyle = '--', alpha=0.5)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$P^{(DSTS)}_{err}-P^{(DTS)}_{err}$')
    plt.tight_layout
    plt.legend()
    plt.show()

    #---------------- PLOT ------------
    fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi=100)
    fig.suptitle(rf'$N={N}$', fontsize=16)
    ax.plot(sigma, abs(p_helstrom_dss-p_dss_hd), linestyle='-', color='k', label='DSS')
    ax.plot(sigma, abs(p_helstrom_cs-p_cs_hd), linestyle='-', color='b', label='CS') 
    ax.legend()
    ax.axvspan(sigma[idx_neg[0]], sigma[idx_neg[-1]], color='blue', alpha=0.10, label = 'Squeezing beneficial')
    ax.axvspan(sigma[idx_neg[-1]], sigma[idx_pos[-1]], color= "#AD0B90", alpha=0.10, label = 'Squeezing not beneficial')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'|$P^{(min)}_{err}-P^{(HD)}_{err}$|')
    plt.legend()
    plt.tight_layout
    plt.show()