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


def theory_point_dss(N:float, beta:float, sigma:float, params:tuple, gauss:tuple):
        
        A, B = params
        x_gh, w_gh = gauss
        a = np.sqrt(N*(1-beta))
        r = np.arcsinh(np.sqrt(N*beta))
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = B*a*np.cos(phi)/np.sqrt(V)
        integral = np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return A*integral


def theory_point_cs(N:float, sigma:float, params:tuple, gauss:tuple):
        
        A, B = params
        x_gh, w_gh = gauss
        a = np.sqrt(N)
        r = 0
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = B*a*np.cos(phi)/np.sqrt(V)
        integral = np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return A*integral


def make_model_dss(sigma, gauss):

        def model_dss(coords, A, B):

            N, beta = coords
            out = np.empty_like(N)
            for i in range(len(N)):
                out[i] = theory_point_dss( N[i], beta[i], sigma, (A, B), gauss)
            return out

        return model_dss


def make_model_cs(sigma, gauss):

        def model_cs(N, A, B):

            out = np.empty_like(N)
            for i in range(len(N)):
                out[i] = theory_point_cs(N[i], sigma, (A, B), gauss)
            return out

        return model_cs



def fit_homodyne_perr(sigmas, print_params=False, cs=False, dss=False, data=False):

    n_gh = 100
    gauss = hermgauss(n_gh)

    fig = go.Figure()
    colors = [ "#e9f063",  "#d62728",  "#2ca02c",  "#ff7f0e",  "#9467bd",  "#17becf",  "#e377c2",  "#8c564b",  "#bcbd22",  "#7f7f7f",  "#5af74b"]*2
    
    #------------------  CS  ------------------
    params_dict_cs = {}

    if print_params:
        print(f"{'='*30}")
        print('=== CS ===')

    for i, sigma in enumerate(sigmas):

        data_cs = np.load(f"data/phase_diff/perr_data_phase_diff_cs_a40_S10000_sigma{sigma}.npz")

        alpha_cs = data_cs["alpha_grid"]
        perr_cs =  data_cs["p_err_cs"]
        sigma_cs = data_cs["sigma"]

        N_cs = alpha_cs**2
        beta_cs = np.linspace(0, 1, len(N_cs))

        x_cs = N_cs
        y_cs = perr_cs

        model_cs = make_model_cs(sigma, gauss)
        params_cs, cov_cs = curve_fit(model_cs, x_cs, y_cs)
        params_err_cs = np.sqrt(np.diag(cov_cs))
        params_dict_cs[f'params_{sigma}'] = params_cs
        params_dict_cs[f'params_err_{sigma}'] = params_err_cs

        if print_params:
            print(f"σ = {sigma}")
            for j, par in enumerate(params_cs):
                print(fr"param_{j}_cs = {par:.3f} ± {params_err_cs[j]:.3f}")
            print(f"{'-'*30}")
        
        N_fit_cs = np.linspace(N_cs.min(), N_cs.max(), 80)
        beta_fit_cs = np.linspace(beta_cs.min(), beta_cs.max(), 80)

        N_surface_cs, beta_surface_cs = np.meshgrid(N_fit_cs, beta_fit_cs, indexing="ij")
        z_surface_cs = np.zeros_like(N_surface_cs)

        for k in range(len(N_fit_cs)):
            for l in range(len(beta_fit_cs)):

                z_surface_cs[k, l] = theory_point_cs(N_fit_cs[k],  sigma_cs, params_cs, gauss)

        if cs:
            # fitted surface
            fig.add_trace(go.Surface(x=N_surface_cs, y=beta_surface_cs, z=z_surface_cs, surfacecolor=np.zeros_like(z_surface_cs), 
                                colorscale=[[0.0, colors[i]], [1.0, colors[i]]], name="Fit", showscale=False))
            
            if data:
                # Simulation scatter extended in beta
                beta_values_cs = np.linspace(0, 1, 20)

                # No beta dependence
                for beta in beta_values_cs:
                    fig.add_trace(go.Scatter3d(x=N_cs, y=np.full_like(N_cs, beta), z=perr_cs, mode="markers", 
                                marker=dict(size=3, color=colors[i]), name=f"Simulation β={beta:.2f}", showlegend=False))

    ####################################################################################################

    # Plotly figure
    #===============================

    colors = ["#8E9C09",  "#6F3434",  "#084108",  "#ff7f0e",  "#460880",  "#0c6f7c",  "#5c0642",  "#941c04",  "#8b8b06",  "#212121",  "#092a06",  "#092a06"]*2

    #------------------  DSS  ------------------
    params_dict_dss = {}

    if print_params:
        print(f"{'='*30}")
        print('=== DSS ===')

    for i, sigma in enumerate(sigmas):

        #===============================
        data_dss = np.load(f"data/phase_diff/perr_data_phase_diff_dss_N40_b40_S10000_sigma{sigma}.npz")
        N_dss =  data_dss["N"]
        beta_dss =  data_dss["beta"]
        perr_dss = data_dss["p_err_dss"]
        sigma_dss = data_dss["sigma"]

        N_mesh_dss, beta_mesh_dss = np.meshgrid(N_dss, beta_dss, indexing="ij")

        x_dss = (N_mesh_dss.ravel(), beta_mesh_dss.ravel())
        y_dss = perr_dss.ravel()

        model_dss = make_model_dss(sigma, gauss)
        params_dss, cov_dss = curve_fit(model_dss,x_dss, y_dss)
        params_err_dss = np.sqrt(np.diag(cov_dss))
        params_dict_dss[f'params_{sigma}'] = params_dss
        params_dict_dss[f'params_err_{sigma}'] = params_err_dss

        if print_params:
            print(f"σ = {sigma}")
            for j, par in enumerate(params_dss):
                print(fr"param_{j}_dss = {par:.3f} ± {params_err_dss[j]:.3f}")
            print(f"{'-'*30}")

        #===============================
        N_fit_dss = np.linspace(N_dss.min(), N_dss.max(), 80)
        beta_fit_dss = np.linspace(beta_dss.min(), beta_dss.max(), 80)

        N_surface_dss, beta_surface_dss = np.meshgrid(N_fit_dss, beta_fit_dss, indexing="ij")
        z_surface_dss = np.zeros_like(N_surface_dss)

        for k in range(len(N_fit_dss)):
            for l in range(len(beta_fit_dss)):

                z_surface_dss[k, l] = theory_point_dss(N_fit_dss[k], beta_fit_dss[l], sigma_dss, params_dss, gauss)

        # fitted surface
        if dss:
            fig.add_trace(go.Surface( x=N_surface_dss, y=beta_surface_dss, z=z_surface_dss, surfacecolor=np.zeros_like(z_surface_dss), 
                                colorscale=[[0.0, colors[i]], [1.0, colors[i]]], name="Fit", showscale=False))
        
            if data:
                fig.add_trace(go.Scatter3d(x=N_mesh_dss.ravel(), y=beta_mesh_dss.ravel(), z=perr_dss.ravel(),
                mode="markers", marker=dict(size=3, color=colors[i]), name=f"σ={sigma}"))

        fig.update_layout(scene=dict(xaxis_title="N", yaxis_title=r"β", zaxis = dict(title="P_err", type="log"), 
                            aspectmode ="cube"), width=900, height=750)

    if cs or dss:
        fig.show()
    return params_dict_cs, params_dict_dss


def beta_threshold_theory(N:float, sigma:float, params_cs:tuple, params_dss:tuple, gauss:tuple):
    '''
    Solves the equation F(β_th) = theory_point_dss - theory_point_cs = 0 for a fixed value of N and σ,
    given the fitted parameters of the models. 
    The search for the root is conducted in the interval (β_min, 1) to ensure that the first root for β=0 is not considered,
    where dF(β_min)/dβ = 0.
    '''
    point_cs = theory_point_cs(N, sigma, params_cs, gauss)
    
    def F(beta):
        
        point_dss = theory_point_dss(N, beta, sigma, params_dss, gauss) 
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


def check_beta_th(N:float, sigma:float, params_cs:tuple, params_dss:tuple):
    '''
    Visualizes the equation F(β) = theory_point_dss - theory_point_cs for a fixed value of N and σ,
    given the fitted parameters of the models. 
    For each (N, σ) pair the β_th is the value for which F(β)=0.
    '''
    n_gh = 100
    gauss = hermgauss(n_gh)

    point_cs = theory_point_cs(N, sigma, params_cs, gauss)
    
    def F(beta):
        
        point_dss = theory_point_dss(N, beta, sigma, params_dss, gauss) 
        return point_dss - point_cs

    beta_grid = np.linspace(0, 1, 500)
    F_b = np.array([F(b) for b in beta_grid])

    idx_min = np.argmin(F_b) 
    beta_min = beta_grid[idx_min]
    beta_max = 1

    print(f"Sign condition: {np.sign(F_b[idx_min]*F(beta_max))}")

    plt.figure(figsize=(6,4))
    plt.plot(beta_grid, F_b)
    plt.axhline(0, color="black")
    plt.axvline(beta_grid[idx_min], linestyle="--", label=F"minimum for N={N}, σ={sigma}")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$F(\beta)=P_{DSS}-P_{CS}$")
    plt.legend()
    plt.show()




def beta_opt_theory(N, sigma, params_dss, gauss):

    def objective(beta):
        return theory_point_dss( N, beta, sigma, params_dss, gauss)

    res = minimize_scalar( objective, bounds=(0,1), method="bounded")
    return res.x

def optimal_squeezing(sigmas, params_dict_cs, params_dict_dss, opt = False, th = False) -> dict:

    colors_th = [ "#1f77b4",  "#d62728",  "#2ca02c",  "#ff7f0e",  "#9467bd",  "#17becf",  "#e377c2",  "#8c564b",  "#bcbd22",  "#7f7f7f", "#5af74b"]*2
    colors_opt = ["#2ca02c","#1f77b4", "#bcbd22","#d62728" ,"#0C3858",  "#6F3434",  "#084108",  "#ff7f0e",  "#460880",  "#0c6f7c",  "#5c0642",  "#941c04",  "#8b8b06",  "#212121", "#092a06"]*2
    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(15,6), dpi=300)

    #---------- FIND THRESHOLD ----------
    N_fit = np.linspace(0, 2, 101)
    beta_fit = np.linspace(0, 1, 101)
    N_surface_cs, beta_surface_cs = np.meshgrid(N_fit, beta_fit, indexing="ij")
    N_surface_dss, beta_surface_dss = np.meshgrid(N_fit, beta_fit, indexing="ij")

    beta_opt_dict = {}

    for i,sigma in enumerate(sigmas):
        params_cs = params_dict_cs[f'params_{sigma}']
        params_dss = params_dict_dss[f'params_{sigma}']
        z_surface_cs = np.zeros_like(N_surface_cs)
        z_surface_dss = np.zeros_like(N_surface_dss)

        for k in range(len(N_fit)):
            for l in range(len(beta_fit)):
                z_surface_cs[k, l] = theory_point_cs(N_fit[k],  sigma, params_cs, gauss)
                z_surface_dss[k, l] = theory_point_dss(N_fit[k], beta_fit[l], sigma, params_dss, gauss)

        # Find intersection points
        difference = z_surface_dss - z_surface_cs
        
        cs = plt.contour(N_surface_cs, beta_surface_cs, difference, levels=[0], alpha=0)
        path = cs.get_paths()[0]
        verts = path.vertices

        N_intersection = verts[:,0]
        beta_intersection = verts[:,1]
        mask = (N_intersection > 0.02) & (beta_intersection > 0.005)

        # Minima along beta for each N
        idx = np.argmin(z_surface_dss, axis=1)   
        beta_opt = beta_fit[idx]
        beta_opt_line = np.zeros_like(N_fit)

        for idx, N in enumerate(N_fit):
            beta_opt_line[idx] = beta_opt_theory(N, sigma, params_dss, gauss)
        beta_opt_dict[f'sigma_{sigma}'] = beta_opt
        
        # Find theoretical values for β_th
        beta_th = []
        N_th = np.linspace(0, 2, 101)
        for N in N_th:
            beta_th.append(beta_threshold_theory(N, sigma, params_cs, params_dss, gauss))
        beta_th = np.array(beta_th)

        if th:
            plt.scatter(N_intersection[mask], beta_intersection[mask], s=30, edgecolors='k', color=colors_th[i], marker='D', zorder=10, label = f'σ={sigma:0.1f}')
            plt.fill_between(N_intersection[mask], beta_intersection[mask], 0, alpha=0.8, zorder=0, color=colors_th[i])
            plt.plot(N_th, beta_th, color='k', linewidth = 3)

        if opt:
            plt.scatter(N_fit, beta_opt, color=colors_opt[i], edgecolors='k', s=50, marker='H', zorder=10, label = f'σ={sigma}')
            plt.plot(N_fit[1:], beta_opt_line[1:], color='k', linewidth = 3)
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



def beta_vs_sigma(N_values:np.array, sigmas:np.array, params_dict_cs:dict, params_dict_dss:dict):

    n_gh = 100
    gauss = hermgauss(n_gh)

    plt.figure(figsize=(8,5), dpi=300)

    for N in N_values:

        beta_th_values = []

        for sigma in sigmas:

            params_cs = params_dict_cs[f'params_{sigma}']
            params_dss = params_dict_dss[f'params_{sigma}']

            beta_th = beta_threshold_theory(N, sigma, params_cs, params_dss, gauss)

            if np.isnan(beta_th):
                beta_th = 0

            beta_th_values.append(beta_th)

        plt.plot(sigmas, beta_th_values, marker="o", label=f"N={N}")
        plt.fill_between(sigmas, beta_th_values, 0, alpha=0.3)

    plt.ylim(-0.01, 1.01)
    plt.xlim(-0.01, 1.01)
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$\beta_{\mathrm{threshold}}$")
    plt.legend()
    plt.grid()
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

def perr_dss_vs_sigma(N:float, beta_opt_dict, sigmas:np.array, params_dict_dss:dict):

    N_grid = np.linspace(0, 2, 81)
    p_dss = np.zeros((len(sigmas)))
    beta_opt_array = np.zeros((len(sigmas)))

    n_gh = 100
    x, w = hermgauss(n_gh)

    for i, sigma in enumerate(sigmas):

        beta_opt_array[i] = beta_opt_dict[f"sigma_{sigma}"][np.where(N_grid == N)[0][0]]

        p_dss[i] = theory_point_dss(N=N, beta = beta_opt_array[i], sigma = sigma, params = params_dict_dss[f'params_{sigma}'], gauss = (x, w))

    return p_dss

def perr_vs_sigma(N:float, beta_opt_dict, sigmas:np.array, params_dict:dict, cs:bool):

    N_grid = np.linspace(0, 2, 81)
    p = np.zeros((len(sigmas)))
    beta_opt_array = np.zeros((len(sigmas)))

    n_gh = 100
    x, w = hermgauss(n_gh)

    if cs:
        for i, sigma in enumerate(sigmas):
        
            p[i] = theory_point_cs(N=N, sigma = sigma, params = params_dict[f'params_{sigma}'], gauss = (x, w))
    else:

        for i, sigma in enumerate(sigmas):
        
            beta_opt_array[i] = beta_opt_dict[f"sigma_{sigma}"][np.where(N_grid == N)[0][0]]
            p[i] = theory_point_dss(N=N, beta = beta_opt_array[i], sigma = sigma, params = params_dict[f'params_{sigma}'], gauss = (x, w))

    return p