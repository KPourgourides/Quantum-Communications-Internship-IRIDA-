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

global DATAPATH_DTS, DATAPATH_DSTS
DATAPATH_DTS = 'data/DTS/perr_dts_N4_mu101_S1000000000'
DATAPATH_DSTS = 'data/DSTS/perr_dsts_N4_b402_mu101_S1000000000'

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


def plot_homodyne_perr(sigmas:list, colors_light:list, colors_dark:list, dts:str|bool = False, dsts:str|bool = False) -> None:
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
        data_dts = np.load(f"{DATAPATH_DTS}_sigma{sigma}.npz")

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
        data_dsts = np.load(f"{DATAPATH_DSTS}_sigma{sigma}.npz")
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

#################################################################################################################
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

def beta_opt_theory(N:float, mu:float, sigma:float, gauss:tuple) -> float:
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


def optimal_squeezing(sigmas:list, colors_opt:list, colors_th:list, opt:bool = False, th:bool = True) -> dict:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for different sigmas.
    '''

    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(15,6), dpi=300)

    #---------- FIND THRESHOLD ----------
    beta_opt_dict = {}

    for i,sigma in enumerate(sigmas):

        # ------------- LOAD DATA ----------------------
               
        data_dts = np.load(f"{DATAPATH_DTS}_sigma{sigma}.npz")
        perr_dts =  data_dts["perr"]

        data_dsts = np.load(f"{DATAPATH_DSTS}_sigma{sigma}.npz")
        perr_dsts =  data_dsts["perr"]
        beta = data_dsts["beta_grid"]
        mus_grid = data_dsts["mus_grid"]
        N = data_dsts["N"]
    
        mus_surface, beta_surface = np.meshgrid(mus_grid, beta, indexing="ij")
        perr_surface_dts = np.zeros_like(mus_surface)

        for k in range(len(mus_grid)):
            for l in range(len(beta)):

                perr_surface_dts [k, l] = perr_dts[k]


        #-------------------------  THRESHOLD  -------------------------
        
        # Find intersection points from two surfaces
        difference = perr_surface_dts - perr_dsts
        
        dts = plt.contour(mus_surface, beta_surface, difference, levels=[0], alpha=0)
        path = dts.get_paths()[0]
        verts = path.vertices

        # sort intersection points and filter out the initial noisy points
        mu_intersection = verts[:,0]
        beta_intersection = verts[:,1]
        mu_intersection, beta_intersection = zip(*sorted(zip(mu_intersection, beta_intersection)))
        mu_intersection = np.array(mu_intersection)
        beta_intersection = np.array(beta_intersection)
        mask = (mu_intersection > 0) & (beta_intersection > 0)
        
        # Find theoretical values for β_th
        
        beta_th = []
        for m in mu_intersection:
            beta_th.append(beta_threshold_theory(N, m, sigma, gauss))
        beta_th= np.array(beta_th)
    
        #-------------------------  R^2  THRESHOLD -------------------------
        
        ss_res_th = np.sum((beta_intersection[mask] - beta_th[mask])**2)
        ss_tot_th = np.sum((beta_intersection[mask] - np.mean(beta_intersection[mask]))**2)
        R2_th = 1 - ss_res_th/ss_tot_th
        
        # ------------- OPTIMAL ----------------------

        # Minima along beta for each N
        valid_rows = ~np.all(np.isnan(perr_dsts), axis=1)
        beta_opt = np.full(len(mus_grid), np.nan)
        idx = np.nanargmin(perr_dsts[valid_rows], axis=1)
        beta_opt[valid_rows] = beta[idx]

        beta_opt_theoretical = np.zeros_like(mus_grid)
        for idx, m in enumerate(mus_grid):
            beta_opt_theoretical[idx] = beta_opt_theory(N, m, sigma, gauss)
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt

        #-------------------------  R^2 OPTIMAL  -------------------------
        ss_res_opt = np.sum((beta_opt[1:] - beta_opt_theoretical[1:])**2)
        ss_tot_opt = np.sum((beta_opt[1:] - np.mean(beta_opt[1:]))**2)
        R2_opt = 1 - ss_res_opt/ss_tot_opt
        

        if th:
            
            plt.scatter(mu_intersection[mask], beta_intersection[mask], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, 
                        label = fr'$\beta_{{\rm th}}$: σ = {sigma}')
            plt.fill_between(mu_intersection[mask], beta_th[mask], 0, alpha=0.8, zorder=0, color=colors_th[i])
            plt.plot(mu_intersection, beta_th, color='k', linewidth = 3)

        if opt:
            plt.scatter(mus_grid[1:], beta_opt[1:], color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, 
                        label = fr'$\beta_{{\rm opt}}$: σ = {sigma}')
            plt.plot(mus_grid[1:], beta_opt_theoretical[1:], color='k', linewidth = 3)
        
        plt.xlabel(r'$\mu$ (Purity of seed state)')
        plt.ylabel(r'$\beta$ (Squeezing Fraction)')

        if th or opt:
            plt.legend()

        plt.tight_layout()
    plt.show()
        
    return beta_opt_dict


def beta_optimal_finder(sigmas:list) -> dict:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for different sigmas.
    '''
    #---------- FIND THRESHOLD ----------
    beta_opt_dict = {}

    for i,sigma in enumerate(sigmas):

        # ------------- LOAD DATA ----------------------
               
        data_dts = np.load(f"{DATAPATH_DTS}_sigma{sigma}.npz")
        perr_dts =  data_dts["perr"]

        data_dsts = np.load(f"{DATAPATH_DSTS}_sigma{sigma}.npz")
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


def beta_threshold_finder(sigmas: list) -> tuple[dict, dict]:
    """
    Finds the threshold β_th by extracting the contour and interpolating
    it onto the fixed mus_grid.
    """

    beta_th_dict = {}
    mus_th_dict = {}

    for sigma in sigmas:

        # ---------- Load data ----------
        data_dts = np.load(f"{DATAPATH_DTS}_sigma{sigma}.npz")
        perr_dts = data_dts["perr"]

        data_dsts = np.load(f"{DATAPATH_DSTS}_sigma{sigma}.npz")
        perr_dsts = data_dsts["perr"]
        beta = data_dsts["beta_grid"]
        mus_grid = data_dsts["mus_grid"]

        mus_surface, beta_surface = np.meshgrid(
            mus_grid, beta, indexing="ij"
        )

        perr_surface_dts = np.repeat(perr_dts[:, None], len(beta), axis=1)

        # ---------- Threshold contour ----------
        difference = perr_surface_dts - perr_dsts

        contour = plt.contour(
            mus_surface,
            beta_surface,
            difference,
            levels=[0],
            alpha=0
        )

        paths = contour.get_paths()

        if len(paths) == 0:
            beta_th_dict[f"sigma_{sigma}"] = np.full_like(mus_grid, np.nan)
            mus_th_dict[f"sigma_{sigma}"] = mus_grid
            continue

        verts = paths[0].vertices

        mu_intersection = verts[:, 0]
        beta_intersection = verts[:, 1]

        # Sort by μ
        order = np.argsort(mu_intersection)
        mu_intersection = mu_intersection[order]
        beta_intersection = beta_intersection[order]

        # Remove duplicate μ values (interp1d requires increasing x)
        mu_unique, idx = np.unique(mu_intersection, return_index=True)
        beta_unique = beta_intersection[idx]

        # Interpolate onto mus_grid
        interp = interp1d(
            mu_unique,
            beta_unique,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        beta_th = interp(mus_grid)

        beta_th_dict[f"sigma_{sigma}"] = beta_th
        mus_th_dict[f"sigma_{sigma}"] = mus_grid

        plt.clf()   # remove invisible contour

    return beta_th_dict
        
    

def sigma_threshold_theory(N:float, mu_grid:float) -> float:

    gauss = hermgauss(100)

    def F(sigma):

        beta_opt = beta_opt_theory(N, mu, sigma, gauss)
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


def sigma_threshold(N, betas, mus) -> float:

    mu_min = 1/(1 + 2*N)
    sigma_list = np.ones((len(mus)))

    for sigma_key in betas.keys():
        if 0 in betas[sigma_key] :
            mu_indices = np.where(betas[sigma_key] == 0)[0]
            sigma_th = float(sigma_key.split('_')[1])
            for l in mu_indices:
                if mus[l] > mu_min:
                    if sigma_list[l] == 1:
                        sigma_list[l] = sigma_th
        
                     
    return sigma_list


