import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import math
from scipy.optimize import curve_fit
from helper_functions.state_measurement import *
from scipy.special import erfc
import plotly.graph_objects as go
from numpy.polynomial.hermite import hermgauss

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


def theory_point_dss(N:float, beta:float, sigma:float, pars:tuple, gauss:tuple):
        
        A, B = pars
        x_gh, w_gh = gauss
        a = np.sqrt(N*(1-beta))
        r = np.arcsinh(np.sqrt(N*beta))
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = B*a*np.cos(phi)/np.sqrt(V)
        integral = np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return A*integral


def theory_point_cs(N:float, sigma:float, pars:tuple, gauss:tuple):
        
        A, B = pars
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
    colors = [ "#1f77b4",  "#d62728",  "#2ca02c",  "#ff7f0e",  "#9467bd",  "#17becf",  "#e377c2",  "#8c564b",  "#bcbd22",  "#7f7f7f"]
    
    #------------------  CS  ------------------

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
        pars_cs, cov_cs = curve_fit(model_cs, x_cs, y_cs)
        pars_err_cs = np.sqrt(np.diag(cov_cs))

        if print_params:
            print(f"σ = {sigma}")
            for j, par in enumerate(pars_cs):
                print(fr"param_{j}_cs = {par:.3f} ± {pars_err_cs[j]:.3f}")
            print(f"{'-'*30}")
        
        N_fit_cs = np.linspace(N_cs.min(), N_cs.max(), 80)
        beta_fit_cs = np.linspace(beta_cs.min(), beta_cs.max(), 80)

        N_surface_cs, beta_surface_cs = np.meshgrid(N_fit_cs, beta_fit_cs, indexing="ij")
        z_surface_cs = np.zeros_like(N_surface_cs)

        for k in range(len(N_fit_cs)):
            for l in range(len(beta_fit_cs)):

                z_surface_cs[k, l] = theory_point_cs(N_fit_cs[k],  sigma_cs, pars_cs, gauss)

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

    colors = ["#0C3858",  "#6F3434",  "#084108",  "#ff7f0e",  "#460880",  "#0c6f7c",  "#5c0642",  "#941c04",  "#8b8b06",  "#212121"]

    #------------------  DSS  ------------------

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
        pars_dss, cov_dss = curve_fit(model_dss,x_dss, y_dss)
        pars_err_dss = np.sqrt(np.diag(cov_dss))

        if print_params:
            print(f"σ = {sigma}")
            for j, par in enumerate(pars_dss):
                print(fr"param_{j}_dss = {par:.3f} ± {pars_err_dss[j]:.3f}")
            print(f"{'-'*30}")

        #===============================
        N_fit_dss = np.linspace(N_dss.min(), N_dss.max(), 80)
        beta_fit_dss = np.linspace(beta_dss.min(), beta_dss.max(), 80)

        N_surface_dss, beta_surface_dss = np.meshgrid(N_fit_dss, beta_fit_dss, indexing="ij")
        z_surface_dss = np.zeros_like(N_surface_dss)

        for k in range(len(N_fit_dss)):
            for l in range(len(beta_fit_dss)):

                z_surface_dss[k, l] = theory_point_dss(N_fit_dss[k], beta_fit_dss[l], sigma_dss, pars_dss, gauss)

        # fitted surface
        if dss:
            fig.add_trace(go.Surface( x=N_surface_dss, y=beta_surface_dss, z=z_surface_dss, surfacecolor=np.zeros_like(z_surface_dss), 
                                colorscale=[[0.0, colors[i]], [1.0, colors[i]]], name="Fit", showscale=False))
        
            if data:
                fig.add_trace(go.Scatter3d(x=N_mesh_dss.ravel(), y=beta_mesh_dss.ravel(), z=perr_dss.ravel(),
                mode="markers", marker=dict(size=3, color=colors[i]), name=f"σ={sigma}"))

        fig.update_layout(scene=dict(xaxis_title="N", yaxis_title=r"β", zaxis = dict(title="P_err", type="log"), 
                            aspectmode ="cube"), width=900, height=750)

    fig.show()