import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from scipy.special import erfc
from helper_functions.state_measurement import measure_coherent

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