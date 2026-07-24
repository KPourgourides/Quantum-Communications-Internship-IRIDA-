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

def perr_cs(alpha_grid:np.array, sigma:float, num_samples:int):
    
    # Calculation of error probability
    #============================================
    p_err = np.zeros((len(alpha_grid)))
    
    for i in range(len(alpha_grid)):

        # Choose phase from Gaussian distribution
        phis = np.random.normal(0, sigma, size=num_samples) 
        wrong_sign_counter = 0

        for phi in phis:
            coherent_sign = np.random.choice([1, -1])
            prog = sf.Program(1)

            with prog.context as q:
                Dgate(coherent_sign*alpha_grid[i]) | q
                Rgate(phi) | q
                MeasureHomodyne(0) | q

            eng = sf.Engine("gaussian")
            result = eng.run(prog)
            result_sign = np.sign(result.samples[0][0])

            if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                    wrong_sign_counter+= 1

        p_err[i] = wrong_sign_counter/num_samples

    return p_err


def perr_dss(N_grid:np.array, beta_grid:np.array, sigma:float, num_samples:int):

    #Calculation of error probability
    #============================================
    p_err = np.zeros((len(N_grid), len(beta_grid)))
    
    for i,N in enumerate(N_grid):
        alphas = np.sqrt(N*(1-beta_grid)) 
        print(f"\rProgress: {i+1}/{len(N_grid)}", end="", flush=True)
        
        for k, beta in enumerate(beta_grid):
            wrong_sign_counter = 0
            r_s = math.asinh(np.sqrt(N*beta))

            # Choose phase from Gaussian distribution
            phis = np.random.normal(0, sigma, size=num_samples)
            for phi in phis:
                coherent_sign = np.random.choice([1, -1])
                prog = sf.Program(1)

                with prog.context as q:
                    Sgate(r_s, 0) | q[0] 
                    Dgate(coherent_sign*alphas[k]) | q
                    Rgate(phi) | q
                    MeasureHomodyne(0) | q

                eng = sf.Engine("gaussian")
                result = eng.run(prog)
                result_sign = np.sign(result.samples[0][0])

                if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                        wrong_sign_counter+= 1

            p_err[i][k] = wrong_sign_counter/num_samples

    return p_err


def theory_point_dss(N:float, beta:float, sigma:float, gauss:tuple):
        
        x_gh, w_gh = gauss
        a = np.sqrt(N*(1-beta))
        r = np.arcsinh(np.sqrt(N*beta))
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = np.sqrt(2)*a*np.cos(phi)/np.sqrt(V)
        integral = np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return 0.5*integral


def theory_point_cs(N:float, sigma:float, gauss:tuple):
        
        x_gh, w_gh = gauss
        a = np.sqrt(N)
        r = 0
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = np.sqrt(2)*a*np.cos(phi)/np.sqrt(V)
        integral = np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return 0.5*integral


def plot_homodyne_perr(sigmas, colors_light, colors_dark, cs=False, dss=False, ):
    '''
    cs and dss can take these values: 'data', 'theory', 'all', False
    '''
    
    n_gh = 100
    gauss = hermgauss(n_gh)

    fig = go.Figure()
    
    #============================  CS  ============================
    
    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------

        data_cs = np.load(f"data/phase_diff/perr_data_phase_diff_cs_a40_S10000_sigma{sigma}.npz")

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

        data_dss = np.load(f"data/phase_diff/perr_data_phase_diff_dss_N40_b40_S10000_sigma{sigma}.npz")
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


def beta_threshold_theory(N:float, sigma:float, gauss:tuple):
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
    beta_grid = np.linspace(0, 1, 500)
    F_b = np.array([F(b) for b in beta_grid])

    beta_min = beta_grid[np.argmin(F_b)]
    beta_max = 1

    try:
        beta_upper = brentq(F, beta_min, beta_max)
        return beta_upper
    
    except:
        return np.nan


def beta_opt_theory(N, sigma, gauss):

    def objective(beta):
        return theory_point_dss( N, beta, sigma, gauss)

    res = minimize_scalar( objective, bounds=(0,1), method="bounded")
    return res.x


def optimal_squeezing(sigmas, colors_opt, colors_th, opt = False, th = False) -> dict:

    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(15,6), dpi=300)

    #---------- FIND THRESHOLD ----------


    beta_opt_dict = {}

    for i,sigma in enumerate(sigmas):

        data_cs = np.load(f"data/phase_diff/perr_data_phase_diff_cs_a40_S10000_sigma{sigma}.npz")
        alpha_cs = data_cs["alpha_grid"]
        perr_cs =  data_cs["p_err_cs"]
        sigma_cs = data_cs["sigma"]
        N_cs = alpha_cs**2
        beta_cs = np.linspace(0, 1, len(N_cs))

        N_surface_cs, beta_surface_cs = np.meshgrid(N_cs, beta_cs, indexing="ij")
        perr_surface_cs = np.zeros_like(N_surface_cs)


        data_dss = np.load(f"data/phase_diff/perr_data_phase_diff_dss_N40_b40_S10000_sigma{sigma}.npz")
        N_dss =  data_dss["N"]
        beta_dss =  data_dss["beta"]
        perr_dss = data_dss["p_err_dss"]
        sigma_dss = data_dss["sigma"]
        N_surface_dss, beta_surface_dss = np.meshgrid(N_dss, beta_dss, indexing="ij")


        for k in range(len(N_cs)):
            for l in range(len(beta_cs)):
                perr_surface_cs [k, l] = perr_cs[k]

        # Find intersection points
        difference = perr_surface_cs - perr_dss
        
        cs = plt.contour(N_surface_cs, beta_surface_cs, difference, levels=[0], alpha=0)
        path = cs.get_paths()[0]
        verts = path.vertices

        N_intersection = verts[:,0]
        beta_intersection = verts[:,1]
        mask = (N_intersection > 0.02) & (beta_intersection > 0.005)

        # Minima along beta for each N
        idx = np.argmin(perr_dss, axis=1)   
        beta_opt = beta_dss[idx]
        beta_opt_line = np.zeros_like(N_dss)

        for idx, N in enumerate(N_dss):
            beta_opt_line[idx] = beta_opt_theory(N, sigma, gauss)
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt
        
        # Find theoretical values for β_th
        beta_th = []
        N_th = np.linspace(0, 2, 101)
        for N in N_th:
            beta_th.append(beta_threshold_theory(N, sigma, gauss))
        beta_th = np.array(beta_th)

        if th:
            plt.scatter(N_intersection[mask], beta_intersection[mask], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, label = f'σ={sigma:0.1f}')
            plt.fill_between(N_intersection[mask], beta_intersection[mask], 0, alpha=0.8, zorder=0, color=colors_th[i])
            plt.plot(N_th, beta_th, color='k', linewidth = 3)

        if opt:
            plt.scatter(N_dss, beta_opt, color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, label = f'σ={sigma}')
            plt.plot(N_dss[1:], beta_opt_line[1:], color='k', linewidth = 3)
            if not th:
                 #plt.ylim(0, 0.5)
                 plt.legend()
        
        plt.xlabel(r'$N$ (Average number of photons)')
        plt.ylabel(r'$\beta$ (Squeezing Fraction)')
        if th:
            plt.legend()

        plt.tight_layout()
    plt.show()
        

    return beta_opt_dict
