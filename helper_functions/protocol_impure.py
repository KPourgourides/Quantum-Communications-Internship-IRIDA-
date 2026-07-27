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
from scipy.optimize import brentq

def perr_dss(N:float, mus_grid:np.array, beta_grid:np.array, sigma:float, num_samples:int, strawberry:bool=False) -> np.array:
        '''
        Experimental calculation of the homodyne error probability for displaced squeezed states
        '''
        # Check beta_grid
        beta_max_grid = mus_grid+(mus_grid-1)/(2*N)
        eng = sf.Engine("gaussian")
        p_err = np.full((len(mus_grid), len(beta_grid)), np.nan)
        mu_min = 1/(1+2*N)
        

        for i, mu in enumerate(mus_grid):
            if mu >= mu_min:
                beta_indices = np.where(beta_grid <= beta_max_grid[i])[0]
                beta_valid = beta_grid[beta_indices]

                alphas = np.sqrt(N*(1-beta_valid)+(mu-1)*(1+2*N*beta_valid)/(2*mu))
                Nth = (1 - mu)/(2*mu)

                print(f"\rProgress: {i+1}/{len(mus_grid)}", end="", flush=True)

                for local_idx, global_idx in enumerate(beta_indices):
        
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
    '''
    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_cs = np.load(f"data/CS/perr_cs_a101_S{int(1e9)}_sigma{sigma}.npz")

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

    '''
    #============================  DSS  ============================

    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_dss = np.load(f"data/DSS/perr_dss_N2_b41_mu41_S50_sigma{sigma}.npz")
        N_dss =  data_dss["N"]
        beta_dss =  data_dss["beta_grid"]
        perr_dss = data_dss["perr"]
        sigma_dss = data_dss["sigma"]
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

        fig.update_layout(scene=dict(xaxis_title="N", yaxis_title=r"β", zaxis = dict(title="P_err", type="log"), 
                            aspectmode ="cube"), width=900, height=750)
        
    if cs or dss:
        fig.show()