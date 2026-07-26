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

            print(f"\rProgress: {i+1}/{len(alpha_grid)}", end="", flush=True)

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

            print(f"\rProgress: {i+1}/{len(N_grid)}", end="", flush=True)

            for k, beta in enumerate(beta_grid):
                p = theory_point_dss(N, beta, sigma, gauss)
                p_err[i, k] = np.random.binomial(num_samples, p) / num_samples
    #============================================
    else:

        eng = sf.Engine("gaussian")

        for i, N in enumerate(N_grid):

            alphas = np.sqrt(N*(1-beta_grid)) 

            print(f"\rProgress: {i+1}/{len(N_grid)}", end="", flush=True)

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


    #============================  DSS  ============================

    for i, sigma in enumerate(sigmas):

        #-------------------------  Load data  -------------------------
        data_dss = np.load(f"data/DSS/perr_dss_N101_b101_S{int(1e9)}_sigma{sigma}.npz")
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


def beta_opt_theory(N:float, sigma:float, gauss:tuple) -> float:
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
        data_cs = np.load(f"data/CS/perr_cs_a101_S{int(1e9)}_sigma{sigma}.npz")
        perr_cs =  data_cs["p_err_cs"]

        data_dss = np.load(f"data/DSS/perr_dss_N101_b101_S{int(1e9)}_sigma{sigma}.npz")
        N =  data_dss["N"]
        beta =  data_dss["beta"]
        perr_dss = data_dss["p_err_dss"]

        N_surface, beta_surface = np.meshgrid(N, beta, indexing="ij")
        perr_surface_cs = np.zeros_like(N_surface)

        for k in range(len(N)):
            for l in range(len(beta)):

                perr_surface_cs [k, l] = perr_cs[k]


        #-------------------------  THRESHOLD  -------------------------
        
        # Find intersection points from two surfaces
        difference = perr_surface_cs - perr_dss
        
        cs = plt.contour(N_surface, beta_surface, difference, levels=[0], alpha=0)
        path = cs.get_paths()[0]
        verts = path.vertices

        # sort intersection points and filter out the initial noisy points
        N_intersection = verts[:,0]
        beta_intersection = verts[:,1]
        N_intersection, beta_intersection = zip(*sorted(zip(N_intersection, beta_intersection)))
        N_intersection = np.array(N_intersection)
        beta_intersection = np.array(beta_intersection)
        mask = (N_intersection > 0.06) & (beta_intersection > 0.01)
        
        # Find theoretical values for β_th
        beta_th = []
        for n in N_intersection:
            beta_th.append(beta_threshold_theory(n, sigma, gauss))
        beta_th= np.array(beta_th)

        #-------------------------  R^2  THRESHOLD -------------------------
        ss_res_th = np.sum((beta_intersection[mask] - beta_th[mask])**2)
        ss_tot_th = np.sum((beta_intersection[mask] - np.mean(beta_intersection[mask]))**2)
        R2_th = 1 - ss_res_th/ss_tot_th

        # ------------- OPTIMAL ----------------------

        # Minima along beta for each N
        idx = np.argmin(perr_dss, axis=1)   
        beta_opt = beta[idx]
        beta_opt_theoretical = np.zeros_like(N)

        for idx, n in enumerate(N):
            beta_opt_theoretical[idx] = beta_opt_theory(n, sigma, gauss)
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt

        #-------------------------  R^2 OPTIMAL  -------------------------
        ss_res_opt = np.sum((beta_opt[1:] - beta_opt_theoretical[1:])**2)
        ss_tot_opt = np.sum((beta_opt[1:] - np.mean(beta_opt[1:]))**2)
        R2_opt = 1 - ss_res_opt/ss_tot_opt


        if th:
            
            plt.scatter(N_intersection[mask], beta_intersection[mask], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, 
                        label = fr'$\beta_{{\rm th}}$: σ = {sigma}, $R^2$ = {R2_th:0.3f}')
            plt.fill_between(N_intersection[mask], beta_th[mask], 0, alpha=0.8, zorder=0, color=colors_th[i])
            plt.plot(N_intersection, beta_th, color='k', linewidth = 3)

        if opt:
            plt.scatter(N[1:], beta_opt[1:], color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, 
                        label = fr'$\beta_{{\rm opt}}$: σ = {sigma}, $R^2$ = {R2_opt:0.3f}')
            plt.plot(N[1:], beta_opt_theoretical[1:], color='k', linewidth = 3)
        
        plt.xlabel(r'$N$ (Average number of photons)')
        plt.ylabel(r'$\beta$ (Squeezing Fraction)')

        if th or opt:
            plt.legend()

        plt.tight_layout()
    plt.show()
        

    return beta_opt_dict


def optimal_squeezing_noisefree(opt:bool = True, th:bool = True) -> None:

    '''
    Plots the threshold and optimal value of the squeezing fraction β as a function of the average photon number for the noisefree case.
    '''

    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(15,6), dpi=300)

    #---------- FIND THRESHOLD ----------

    data_cs = np.load(f"data/CS/perr_cs_a101_S{int(1e9)}_sigma0.0.npz")
    perr_cs =  data_cs["p_err_cs"]

    data_dss = np.load(f"data/DSS/perr_dss_N101_b101_S{int(1e9)}_sigma0.0.npz")
    N =  data_dss["N"]
    beta =  data_dss["beta"]
    perr_dss = data_dss["p_err_dss"]
    N_surface, beta_surface = np.meshgrid(N, beta, indexing="ij")

    beta_cs = np.linspace(0, 1, len(N))
    N_surface_cs, beta_surface_cs = np.meshgrid(N, beta_cs, indexing="ij")
    perr_surface_cs = np.zeros_like(N_surface_cs)

    for k in range(len(N)):
        for l in range(len(beta_cs)):

            perr_surface_cs [k, l] = perr_cs[k]

    # Find intersection points
    difference = perr_surface_cs - perr_dss
        
    cs = plt.contour(N_surface_cs, beta_surface_cs, difference, levels=[0], alpha=0)
    path = cs.get_paths()[0]
    verts = path.vertices

    N_intersection = verts[:,0]
    beta_intersection = verts[:,1]
    N_intersection, beta_intersection = zip(*sorted(zip(N_intersection, beta_intersection)))
    N_intersection = np.array(N_intersection)
    beta_intersection = np.array(beta_intersection)
    mask = (N_intersection > 0.01) & (beta_intersection > 0.01)

        
    # ------------- THRESHOLD ----------------------
    beta_th = 4*N_intersection/(4*N_intersection + 1)


    #-------------------------  R^2 THRESHOLD  -------------------------
    ss_res_th = np.sum((beta_intersection[mask] - beta_th[mask])**2)
    ss_tot_th = np.sum((beta_intersection[mask] - np.mean(beta_intersection[mask]))**2)
    R2_th = 1 - ss_res_th/ss_tot_th


    # ------------- OPTIMAL ----------------------
    # Minima along beta for each N
    idx = np.argmin(perr_dss, axis=1)   
    beta_opt = beta[idx]
    beta_opt_line =  N/(2*N + 1)

    #-------------------------  R^2 OPTIMAL  -------------------------
    ss_res_opt = np.sum((beta_opt - beta_opt_line)**2)
    ss_tot_opt = np.sum((beta_opt - np.mean(beta_opt))**2)
    R2_opt = 1 - ss_res_opt/ss_tot_opt

    if th:
            
        plt.scatter(N_intersection[mask], beta_intersection[mask], s=30, edgecolors='k', color='white', marker='D', zorder=10, 
                    label = fr'$\beta_{{\rm th}}$: $R^2$ = {R2_th:0.3f}')
        plt.fill_between(N_intersection[mask], beta_th[mask], 0, alpha=0.8, zorder=0, color='blue')
        plt.plot(N_intersection, beta_th, color='k', linewidth = 3)

    if opt:
        plt.scatter(N[1:], beta_opt[1:], color='red', edgecolors='k', s=50, marker='o', zorder=10, 
                    label = fr'$\beta_{{\rm opt}}$: $R^2$ = {R2_opt:0.3f}')
        plt.plot(N[1:], beta_opt_line[1:], color='k', linewidth = 3)

        if not th:
             plt.legend()
        
    plt.xlabel(r'$N$ (Average number of photons)')
    plt.ylabel(r'$\beta$ (Squeezing Fraction)')
    if th:
        plt.legend()

    plt.tight_layout()
    plt.show()


def beta_threshold_vs_sigma(N_values:np.array, sigmas:np.array):

    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(5,5))

    for N in N_values:

        beta_th_values = []

        for sigma in sigmas:

            beta_th = beta_threshold_theory(N, sigma, gauss)

            if np.isnan(beta_th):
                beta_th = 0

            beta_th_values.append(beta_th)

        plt.plot(sigmas, beta_th_values, label=f"N={N}", linewidth=3)
        plt.fill_between(sigmas, beta_th_values, 0, alpha=0.3)

    plt.ylim(-0.01, 1.01)
    plt.xlim(-0.01, 1.01)
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$\beta_{\mathrm{threshold}}$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def helstrom_bound(N:float, beta_opt_dict:dict, sigmas:np.array, fock_cutoff:int):

    N_grid = np.linspace(0, 2, 81)

    if beta_opt_dict == 0:
        beta_opt_array = np.zeros((len(sigmas)))
    else:
        beta_opt_array = np.zeros((len(sigmas)))

        for i,sigma in enumerate(sigmas):

            beta_opt_array[i] = beta_opt_dict[f"sigma_{sigma}"][np.where(N_grid == N)[0][0]]


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
    
    n_gh = 100
    gauss = hermgauss(n_gh)
    x, w = gauss
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

def perr_vs_sigma(N:float, beta_opt_dict, sigmas:np.array, cs:bool):

    N_grid = np.linspace(0, 2, 101)
    p = np.zeros((len(sigmas)))
    beta_opt_array = np.zeros((len(sigmas)))

    n_gh = 100
    x, w = hermgauss(n_gh)

    if cs:
        for i, sigma in enumerate(sigmas):
        
            p[i] = theory_point_cs(N=N, sigma = sigma, gauss = (x, w))
    else:

        for i, sigma in enumerate(sigmas):
        
            beta_opt_array[i] = beta_opt_dict[f"sigma_{sigma}"][np.where(N_grid == N)[0][0]]
            p[i] = theory_point_dss(N=N, beta = beta_opt_array[i], sigma = sigma, gauss = (x, w))

    return p