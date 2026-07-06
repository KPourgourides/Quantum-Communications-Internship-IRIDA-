import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import math
from scipy.special import erfc
from helper_functions.state_measurement import *
from scipy.special import erfc


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


def P_err_theory(N, beta):
    """
    Theoretical homodyne error probability for DSS states.

    Parameters
    ----------
    N : float or np.ndarray
        Total photon number
    beta : float or np.ndarray
        Squeezing fraction (0 <= beta <= 1)

    Returns
    -------
    P_err : float or np.ndarray
    """

    alpha = np.sqrt(N * (1 - beta))
    Sigma = 1.0 / (N * beta + np.sqrt(1 + N * beta))

    argument = (2 * alpha) / (np.sqrt(2) * Sigma)

    return 0.5 * erfc(argument)