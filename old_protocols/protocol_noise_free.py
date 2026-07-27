import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import math
from scipy.special import erfc
from scipy.optimize import curve_fit
from helper_functions.state_measurement import *
from scipy.special import erfc
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def perr_cs(alpha_grid:np.array, homodyne_angle:float, num_samples:int):

    #Calculation of error probability
    #============================================
    p_err = np.zeros(len(alpha_grid))

    for i,alpha in enumerate(alpha_grid):

        print(f"\rProgress: {i+1}/{len(alpha_grid)}", end="", flush=True)

        wrong_sign_counter = 0

        for j in range(num_samples):

            coherent_sign = np.random.choice([1, -1])
            result = measure_coherent(coherent_sign*alpha, homodyne_angle=homodyne_angle)
            result_sign = np.sign(result)

            if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                    wrong_sign_counter+= 1

        p_err[i] = wrong_sign_counter/num_samples
    #============================================

    N = np.abs(alpha_grid)**2

    return N, p_err


def perr_dss(N_grid:np.array, beta_grid:np.array, homodyne_angle:float, num_samples:int):

    #Calculation of error probability
    #============================================
    p_err = np.zeros((len(N_grid), len(beta_grid)))
    
    for i,N in enumerate(N_grid):
        print(f"\rProgress: {i+1}/{len(N_grid)}", end="", flush=True)
        alpha_grid = np.sqrt(N*(1-beta_grid)) 
        
        for k, beta in enumerate(beta_grid):
            wrong_sign_counter = 0
            r_s = math.asinh(np.sqrt(N*beta))
            
            for j in range(num_samples):    
                coherent_sign = np.random.choice([1, -1])
                
                result = measure_displaced_squeezed(r_s, 0, coherent_sign*alpha_grid[k], homodyne_angle)
                result_sign = np.sign(result)
    
                if (result_sign>=0 and coherent_sign<0) or (result_sign<0 and coherent_sign>0):
                        wrong_sign_counter+= 1
    
            p_err[i][k] = wrong_sign_counter/num_samples

    return p_err

'''
def perr_homodyne(N, beta):
     
     # Error probability for homodyne detection in the general case. For CS, beta=0
    alpha = np.sqrt(N * (1 - beta))
    Sigma = 1/(np.sqrt(N*beta) + np.sqrt(1+ N*beta))
    z = 2*alpha/(np.sqrt(2)*Sigma)
    perr = 0.5*erfc(z) 
    
    return perr
'''

def helstrom_bound(N, beta):

    # Minimum theoretical error probability from the Helstrom bound
    exponent = -4 * N * (1 - beta) * (1 + 2 * N * beta + 2 * np.sqrt(N * beta * (1 + N * beta)))
    
    perr = 0.5 * (1 - np.sqrt(1 - np.exp(exponent)))

    return perr


def model_dss(coords, A, B):
        
        N, beta = coords
        alpha = np.sqrt(N*(1-beta))
        Sigma = 1 / (np.sqrt(N*beta) +np.sqrt(1 + N*beta))

        return (A * erfc(B*alpha/Sigma)).ravel()


def model_cs(coords, A, B):
        
        N, beta = coords
        return A * erfc(B*np.sqrt(N))


def fit_homodyne_perr(N_cs, beta_cs, N_dss, beta_dss, perr_cs, perr_dss, dss=True, cs=True, data=False):

    #------------------  CS  ------------------
    # Fit the data to the homodyne theory
    z_data_cs = perr_cs
    params_cs, covariance_cs = curve_fit(model_cs, (N_cs, beta_cs), z_data_cs)
    params_err_cs = np.sqrt(np.diag(covariance_cs))

    print('--- CS ---')
    for i, param in enumerate(params_cs):
        print(fr"param_{i}_cs = {param:.3f} ± {params_err_cs[i]:.3f}")


    # Create fitted surface
    N_fit_cs = np.linspace(N_cs.min(), N_cs.max(), 200)
    beta_fit_cs = np.linspace(0, 1, 200)
    N_surface_cs, beta_surface_cs = np.meshgrid(N_fit_cs, beta_fit_cs, indexing="ij")
    z_surface_cs = model_cs((N_surface_cs, beta_surface_cs), *params_cs)
    
    # Plot
    fig = go.Figure()
    if cs:
        fig.add_trace(go.Surface(x=N_surface_cs, y=beta_surface_cs, z=z_surface_cs, colorscale="Blues_r", name="Fitted surface", showscale=False))

        if data:
            # Simulation scatter extended in beta
            beta_values_cs = np.linspace(0, 1, 20)

            # No beta dependence
            for beta in beta_values_cs:
                fig.add_trace(go.Scatter3d(x=N_cs, y=np.full_like(N_cs, beta), z=z_data_cs, mode="markers", 
                            marker=dict(size=3, color=z_data_cs, colorscale="Viridis"), name=f"Simulation β={beta:.2f}", showlegend=False))
               
    #------------------ DSS ------------------
    N_mesh_dss, beta_mesh_dss = np.meshgrid(N_dss, beta_dss, indexing="ij")

    # Fit data to homodyne theory
    z_data_dss = perr_dss.ravel()
    params_dss, covariance_dss = curve_fit(model_dss,(N_mesh_dss, beta_mesh_dss), z_data_dss)
    params_err_dss = np.sqrt(np.diag(covariance_dss))

    print('--- DSS ---')
    for i, param in enumerate(params_dss):
        print(fr"param_{i}_dss = {param:.3f} ± {params_err_dss[i]:.3f}")

    # Create fitted surface
    N_fit_dss = np.linspace(N_mesh_dss.min(), N_mesh_dss.max(), 200)
    beta_fit_dss = np.linspace(beta_mesh_dss.min(), beta_mesh_dss.max(), 200)
    N_surface_dss, beta_surface_dss = np.meshgrid(N_fit_dss, beta_fit_dss, indexing="ij")
    z_surface_dss = model_dss((N_surface_dss, beta_surface_dss), *params_dss).reshape(N_surface_dss.shape)
    
    if dss:
        fig.add_trace(go.Surface(x=N_surface_dss, y=beta_surface_dss, z=z_surface_dss, colorscale="Reds_r", opacity=1, name="Fit", showscale=False))

        if data:
            fig.add_trace(go.Scatter3d(x=N_mesh_dss.ravel(), y=beta_mesh_dss.ravel(), z=z_data_dss.ravel(), mode='markers', 
                        marker=dict(size=3, color=z_data_dss.ravel(), colorscale='Viridis', opacity=1), name="Simulation"))

    #figure settings
    fig.update_layout(title="Fit: State Discrimination Error Probability (Homodyne Detection)", width=800, height=800, 
                    scene=dict(xaxis_title="N", yaxis_title="β", zaxis = dict(title="P_err", type="log"), aspectmode="cube"))
    
    if cs or dss:
        fig.show()

    return params_cs, params_err_cs, params_dss, params_err_dss


def beta_th(N, a, b):
    return a*N/(b*N + 1)


def beta_opt(N, a, b):
    return a*N/(b*N + 1)


def optimal_squeezing(params_cs, params_dss):

    #---------- FIND THRESHOLD ----------

    # Find intersection points
    N = np.linspace(0, 2, 100)
    beta = np.linspace(0, 1, 100)
    N_grid, beta_grid = np.meshgrid(N, beta, indexing="ij")
    Z1 = model_dss((N_grid, beta_grid), *params_dss).reshape(N_grid.shape)
    Z2 = model_cs((N_grid, beta_grid), *params_cs)
    difference = Z1 - Z2

    plt.figure(figsize=(10,4), dpi=300)
    cs = plt.contour(N_grid, beta_grid, difference, levels=[0], alpha=0)
    path = cs.get_paths()[0]
    verts = path.vertices

    N_intersection = verts[:,0]
    beta_intersection = verts[:,1]

    # Fit intersection points to theoretical curve
    pars_th, pcov_th = curve_fit(beta_th, N_intersection, beta_intersection)
    pars_th_err = np.sqrt(np.diag(pcov_th))

    print('--- THRESHOLD ---')
    print(f"A_th = {pars_th[0]:.3f} ± {pars_th_err[0]:.3f}, {(np.abs(pars_th[0] - 4)/(pars_th_err[0])):.3f}σ away from theoretical value")
    print(f"B_th = {pars_th[1]:.3f} ± {pars_th_err[1]:.3f}, {(np.abs(pars_th[1] - 4)/(pars_th_err[1])):.3f}σ away from theoretical value")

    #---------- FIND OPTIMAL ----------

    N_mesh, beta_mesh = np.meshgrid(N_grid, beta_grid, indexing="ij")

    # Generate fitted surface to fnd minima
    N_fit = np.linspace(N_mesh.min(), N_mesh.max(), 100)
    beta_fit = np.linspace(beta_mesh.min(), beta_mesh.max(), 100)
    N_surface, beta_surface = np.meshgrid(N_fit, beta_fit, indexing="ij")

    # Fitted surface
    z_surface = model_dss((N_surface, beta_surface),*params_dss).reshape(N_surface.shape)

    # Minima along beta for each N
    idx = np.argmin(z_surface, axis=1)   
    beta_min = beta_fit[idx]
    
    # Fit minima to theoretical curve
    params_opt, pcov_opt = curve_fit(beta_opt, N_fit, beta_min)
    params_opt_err = np.sqrt(np.diag(pcov_opt))

    print('--- OPTIMAL ---')
    print(f"A_opt = {params_opt[0]:.3f} ± {params_opt_err[0]:.3f}, {(np.abs(params_opt[0] - 1)/(params_opt_err[0])):.3f}σ away from theoretical value")
    print(f"B_opt = {params_opt[1]:.3f} ± {params_opt_err[1]:.3f}, {(np.abs(params_opt[1] - 2)/(params_opt_err[1])):.3f}σ away from theoretical value")

    # Plot
    
    # Intersection
    plt.scatter(N_intersection, beta_intersection, s=10, color='r', edgecolors='k', marker='D', zorder=10)
    plt.fill_between(N, beta_th(N, *(pars_th-pars_th_err)), beta_th(N, *(pars_th+pars_th_err)), alpha=0.5, color='gray', zorder=5)
    plt.plot(N, beta_th(N, *pars_th), color='red', linewidth=1, label = rf'$\beta_{{\rm th}}(N)=\frac{{{pars_th[0]:.2f}\,N}}{{{pars_th[1]:.2f}\,N+1}}$', zorder=5)
    plt.fill_between(N, beta_th(N, *pars_th), 0, color='blue', alpha=0.2)
    plt.fill_between(N, beta_th(N, *pars_th), 1, color='red', alpha=0.2)

    # Minima
    plt.fill_between(N, beta_opt(N, *(params_opt-params_opt_err)), beta_opt(N, *(params_opt+params_opt_err)), alpha=0.5, color='gray')
    plt.plot(N, beta_th(N, *params_opt), color='Blue', linewidth=1, label = rf'$\beta_{{\rm opt}}(N)=\frac{{{params_opt[0]:.2f}\,N}}{{{params_opt[1]:.2f}\,N+1}}$', zorder=1)
    plt.scatter(N_fit, beta_min, color='blue', edgecolors='k', s=10, marker='D', zorder=10)

    # Texts
    plt.text(x=0.5, y=0.9, s="Coherent States are easier to discriminate", color='red', fontweight='bold')
    plt.text(x=0.8, y=0.6, s="Displaced Squeezed States are easier to discriminate", color='blue', fontweight='bold')

    plt.xlabel(r'$N$ (Average number of photons)')
    plt.ylabel(r'$\beta$ (Squeezing Fraction)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return params_opt, params_opt_err
    


     

        