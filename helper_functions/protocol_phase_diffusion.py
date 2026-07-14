import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import math
from scipy.optimize import curve_fit
from helper_functions.state_measurement import *
from scipy.special import erfc
import plotly.graph_objects as go
from numpy.polynomial.hermite import hermgauss

def phase_diffused_cs(alpha_grid: np.array, sigma: float, num_samples: int):
    
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

        
def phase_diffused_dss(N_grid:np.array, beta_grid:np.array, sigma: float, num_samples:int):

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


def fit_homodyne_perr(sigmas, print_err=False):

    # Compute once
    n_gh = 100
    x_gh, w_gh = hermgauss(n_gh)

    def theory_point(N, beta, sigma, A, B):

        a = np.sqrt(N*(1-beta))
        r = np.arcsinh(np.sqrt(N*beta))
        phi = np.sqrt(2) * sigma * x_gh
        V = (np.exp(-2*r)*np.cos(phi)**2 + np.exp(2*r)*np.sin(phi)**2)

        arg = B*a*np.cos(phi)/np.sqrt(V)
        integral = np.sum(w_gh * erfc(arg)) / np.sqrt(np.pi)

        return A * integral

    def make_model(sigma):

        def model(coords, A, B):

            N, beta = coords
            out = np.empty_like(N)
            for i in range(len(N)):
                out[i] = theory_point( N[i], beta[i], sigma=sigma, A=A, B=B)
            return out

        return model


    # Plotly figure
    #===============================

    fig = go.Figure()
    colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
    ]
    
    for i, sigma in enumerate(sigmas):

        #===============================
        data = np.load(f"data/perr_data_phase_diff_dss_N40_b40_S10000_sigma{sigma}.npz")
        N_dss = data["N"]
        beta_dss = data["beta"]
        p_err_dss = data["p_err_dss"]
        sigma_dss = data["sigma"]

        N_mesh, beta_mesh = np.meshgrid(N_dss, beta_dss, indexing="ij")

        x = (N_mesh.ravel(), beta_mesh.ravel())
        y = p_err_dss.ravel()

        model = make_model(sigma)
        pars, cov = curve_fit(model,x, y)
        A_fit, B_fit = pars
        A_err, B_err = np.sqrt(np.diag(cov))

        if print_err:
            print(f"σ = {sigma}")
            print(fr"A_cs = {A_fit:.3f} ± {A_err:.3f}, {(np.abs(A_fit - 0.5)/(A_err)):.3f}σ away from theoretical value")
            print(f"B_cs = {B_fit:.3f} ± {B_err:.3f}, {(np.abs(B_fit - np.sqrt(2))/(B_err)):.3f}σ away from theoretical value")
            print(f"{'-'*30}")

        #===============================
        N_plot = np.linspace(N_dss.min(), N_dss.max(), 80)
        beta_plot = np.linspace(beta_dss.min(), beta_dss.max(), 80)

        N_surface, beta_surface = np.meshgrid(N_plot, beta_plot, indexing="ij")

        P_surface = np.zeros_like(N_surface)

        for k in range(len(N_plot)):
            for j in range(len(beta_plot)):

                P_surface[k, j] = theory_point(N_plot[k], beta_plot[j], sigma=sigma_dss, A=A_fit, B=B_fit)

        # fitted surface
        fig.add_trace(
            go.Surface( x=N_surface, y=beta_surface, z=P_surface, 
                       
                        # Constant value everywhere
                        surfacecolor=np.zeros_like(P_surface),
                        # Same color at both ends
                        colorscale=[[0.0, colors[i]], [1.0, colors[i]]],

                        opacity=1, name="Fit", showscale=False)
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="N",
                yaxis_title=r"β",
                zaxis = dict(title="P_err", type="log"),
                aspectmode ="cube",
            ),
            width=900,
            height=750,
        )

    fig.show()