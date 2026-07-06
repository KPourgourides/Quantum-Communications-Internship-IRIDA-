import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from scipy.special import erfc

def perr_cs(alpha_grid:any, homodyne_angle:float, num_samples:int):

    def coherent_states(alpha:complex, homodyne_angle:float):

        prog = sf.Program(1)
        r = np.abs(alpha)
        phi = np.angle(alpha)

        with prog.context as q:
            Coherent(r, phi) | q[0]
            MeasureHomodyne(homodyne_angle) | q[0]

        #run the engine and get the state
        eng = sf.Engine("gaussian")
        result = eng.run(prog)

        return result.samples[0, 0]

    p_err = np.zeros(len(alpha_grid))

    for i,alpha in enumerate(alpha_grid):

        print(f"\rProgress: {i+1}/{len(alpha_grid)}", end="", flush=True)

        wrong_sign_counter = 0

        for j in range(num_samples):

            sign = np.random.choice([1, -1])

            result = coherent_states(sign*alpha, homodyne_angle=homodyne_angle)

            if np.sign(result) != sign:

                    wrong_sign_counter+= 1

        p_err[i] = wrong_sign_counter/num_samples

    N = np.abs(alpha_grid)**2

    return N, p_err