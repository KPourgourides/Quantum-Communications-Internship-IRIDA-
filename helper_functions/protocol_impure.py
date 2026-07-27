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

def perr_dss(N:float, mus_grid:np.array, beta_grid:np.array, sigma:float, num_samples:int) -> np.array:
        '''
        Experimental calculation of the homodyne error probability for displaced squeezed states
        '''
        # Check beta_grid
        eng = sf.Engine("gaussian")
        p_err = np.full((len(mus_grid), len(beta_grid)), np.nan)
        mu_min = 1/(1+2*N)
        

        for i, mu in enumerate(mus_grid):
            if mu >= mu_min:

                alphas2 = N*(1-beta_grid)+(mu-1)*(1+2*N*beta_grid)/(2*mu)
                alpha_valid_indices = np.where(alphas2 >= 0)[0]
                alphas = np.sqrt(alphas2[alpha_valid_indices])
                Nth = (1 - mu)/(2*mu)
                
                print(f"\rProgress: {i+1}/{len(mus_grid)}", end="", flush=True)

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


def perr_cs(N:float, mus_grid:np.array, sigma:float, num_samples:int) -> np.array:
        '''
        Experimental calculation of the homodyne error probability for coherent states
        '''
        # Check beta_grid
        eng = sf.Engine("gaussian")
        p_err = np.full((len(mus_grid)), np.nan)
        mu_min = 1/(1+2*N)
        

        for i, mu in enumerate(mus_grid):
            if mu >= mu_min:

                alpha = np.sqrt(N+(mu-1)/(2*mu))
                Nth = (1 - mu)/(2*mu)
                
                print(f"\rProgress: {i+1}/{len(mus_grid)}", end="", flush=True)

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
        data_cs = np.load(f"data/CS/perr_cs_N2_mu21_S10000_sigma{sigma}.npz")

        N_cs = data_cs["N"]
        perr_cs =  data_cs["perr"]
        mus_cs = data_cs["mus_grid"]
        beta_cs = np.linspace(0, 1, len(mus_cs))
        
        N_surface_cs, beta_surface_cs = np.meshgrid(mus_cs, beta_cs, indexing="ij")
        z_surface_cs = np.zeros_like(N_surface_cs)
        perr_surface_cs = np.zeros_like(N_surface_cs)

        #-------------------------  Theoretical curve  -------------------------
        
        for k in range(len(beta_cs)):
            for l in range(len(beta_cs)):

                perr_surface_cs [k, l] = perr_cs[k]
                #z_surface_cs[k, l] = theory_point_cs(N_cs[k],  sigma_cs, gauss)
        
        #-------------------------  R^2  -------------------------
        '''
        ss_res_cs = np.sum((perr_cs - z_surface_cs[:,0])**2)
        ss_tot_cs = np.sum((perr_cs - np.mean(perr_cs))**2)
        R2_cs = 1 - ss_res_cs/ss_tot_cs
        '''
        #-------------------------  Plot  -------------------------
        '''
        if cs in ['theory', 'all']:
        
            fig.add_trace(go.Surface(x=N_surface_cs, y=beta_surface_cs, z=z_surface_cs, surfacecolor=np.zeros_like(z_surface_cs), 
                                colorscale=[[0.0, colors_light[i]], [1.0, colors_light[i]]], showscale=False))
        '''
        if cs in ['data', 'all']:
            
            fig.add_trace(go.Scatter3d(x=N_surface_cs.ravel(), y=beta_surface_cs.ravel(), z=perr_surface_cs.ravel(), mode="markers", 
                                marker=dict(size=3, color=colors_dark[i])))
            fig.add_trace(go.Surface(x=N_surface_cs, y=beta_surface_cs, z=perr_surface_cs, colorscale=[[0.0, colors_light[i]], [1.0, colors_light[i]]], showscale=False))

    
    #============================  DSS  ============================

    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_dss = np.load(f"data/DSS/perr_dss_N2_b21_mu21_S10000_sigma{sigma}.npz")
        N_dss =  data_dss["N"]
        beta_dss =  data_dss["beta_grid"]
        perr_dss = data_dss["perr"]
        mus_dss = data_dss["mus_grid"]

        mu_surface_dss, beta_surface_dss = np.meshgrid(mus_dss, beta_dss, indexing="ij")
        

        #-------------------------  Theoretical curve  -------------------------
        '''
        z_surface_dss = np.zeros_like(mus_surface_dss)

        for k in range(len(N_dss)):
            for l in range(len(beta_dss)):

                z_surface_dss[k, l] = theory_point_dss(N_dss[k], beta_dss[l], sigma_dss, gauss)
        '''
        #-------------------------  R^2  -------------------------
        '''
        ss_res_dss = np.sum((perr_dss - z_surface_dss)**2)
        ss_tot_dss = np.sum((perr_dss - np.mean(perr_dss))**2)
        R2_dss = 1 - ss_res_dss/ss_tot_dss
        '''
        #-------------------------  Plot  -------------------------
        '''
        if dss in ['theory', 'all']:
                fig.add_trace(go.Surface( x=N_surface_dss, y=beta_surface_dss, z=z_surface_dss, surfacecolor=np.zeros_like(z_surface_dss), 
                                colorscale=[[0.0, colors_dark[i]], [1.0, colors_dark[i]]], showscale=False))
        '''
        if dss in ['data', 'all']:
                fig.add_trace(go.Scatter3d(x=mu_surface_dss.ravel(), y=beta_surface_dss.ravel(), z=perr_dss.ravel(),
                mode="markers", marker=dict(size=3, color=colors_light[i])))
                fig.add_trace(go.Surface(x=mu_surface_dss, y=beta_surface_dss, z=perr_dss, colorscale=[[0.0, colors_dark[i]], [1.0, colors_dark[i]]], showscale=False))

        fig.update_layout(scene=dict(xaxis_title="μ", yaxis_title=r"β", zaxis = dict(title="P_err", type="log"), 
                            aspectmode ="cube"), width=900, height=750)
        
    if cs or dss:
        fig.show()

#################################################################################################################


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
               
        data_cs = np.load(f"data/CS/perr_cs_N2_mu21_S10000_sigma{sigma}.npz")
        perr_cs =  data_cs["perr"]

        data_dss = np.load(f"data/DSS/perr_dss_N2_b21_mu21_S10000_sigma{sigma}.npz")
        perr_dss =  data_dss["perr"]
        beta = data_dss["beta_grid"]
        mus_grid = data_dss["mus_grid"]
    
        mus_surface, beta_surface = np.meshgrid(mus_grid, beta, indexing="ij")
        perr_surface_cs = np.zeros_like(mus_surface)

        for k in range(len(mus_grid)):
            for l in range(len(beta)):

                perr_surface_cs [k, l] = perr_cs[k]


        #-------------------------  THRESHOLD  -------------------------
        
        # Find intersection points from two surfaces
        difference = perr_surface_cs - perr_dss
        
        cs = plt.contour(mus_surface, beta_surface, difference, levels=[0], alpha=0)
        path = cs.get_paths()[0]
        verts = path.vertices

        # sort intersection points and filter out the initial noisy points
        mu_intersection = verts[:,0]
        beta_intersection = verts[:,1]
        mu_intersection, beta_intersection = zip(*sorted(zip(mu_intersection, beta_intersection)))
        mu_intersection = np.array(mu_intersection)
        beta_intersection = np.array(beta_intersection)
        mask = (mu_intersection > 0.06) & (beta_intersection > 0.01)
        
        # Find theoretical values for β_th
        '''
        beta_th = []
        for n in N_intersection:
            beta_th.append(beta_threshold_theory(n, sigma, gauss))
        beta_th= np.array(beta_th)
        '''
        #-------------------------  R^2  THRESHOLD -------------------------
        '''
        ss_res_th = np.sum((beta_intersection[mask] - beta_th[mask])**2)
        ss_tot_th = np.sum((beta_intersection[mask] - np.mean(beta_intersection[mask]))**2)
        R2_th = 1 - ss_res_th/ss_tot_th
        '''
        # ------------- OPTIMAL ----------------------

        # Minima along beta for each N
        idx = np.argmin(perr_dss, axis=1)   
        beta_opt = beta[idx]
        #beta_opt_theoretical = np.zeros_like(N)

        '''
        for idx, n in enumerate(N):
            beta_opt_theoretical[idx] = beta_opt_theory(n, sigma, gauss)
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt

        #-------------------------  R^2 OPTIMAL  -------------------------
        ss_res_opt = np.sum((beta_opt[1:] - beta_opt_theoretical[1:])**2)
        ss_tot_opt = np.sum((beta_opt[1:] - np.mean(beta_opt[1:]))**2)
        R2_opt = 1 - ss_res_opt/ss_tot_opt
        '''

        if th:
            
            plt.scatter(mu_intersection[mask], beta_intersection[mask], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, 
                        label = fr'$\beta_{{\rm th}}$: σ = {sigma}')
            #plt.fill_between(N_intersection[mask], beta_th[mask], 0, alpha=0.8, zorder=0, color=colors_th[i])
            #plt.plot(N_intersection, beta_th, color='k', linewidth = 3)

        if opt:
            plt.scatter(mus_grid[1:], beta_opt[1:], color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, 
                        label = fr'$\beta_{{\rm opt}}$: σ = {sigma}')
            #plt.plot(N[1:], beta_opt_theoretical[1:], color='k', linewidth = 3)
        
        plt.xlabel(r'$\mu$ (Purity of seed state)')
        plt.ylabel(r'$\beta$ (Squeezing Fraction)')

        if th or opt:
            plt.legend()

        plt.tight_layout()
    plt.show()
        

    return beta_opt_dict