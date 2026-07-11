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