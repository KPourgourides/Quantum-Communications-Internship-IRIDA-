import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *


def measure_fock(n):
    prog = sf.Program(1)

    with prog.context as q:
        Fock(n) | q[0]
        MeasureFock() | q[0]

    #run the engine and get the state
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 100})

    result = eng.run(prog)

    return result.samples


def measure_coherent(alpha, homodyne_angle):

    prog = sf.Program(1)
    r = np.abs(alpha)
    phi = np.angle(alpha)
    
    with prog.context as q:
        Coherent(r, phi) | q[0]
        MeasureHomodyne(homodyne_angle) | q[0]

    #run the engine and get the state
    eng = sf.Engine("gaussian")
    result = eng.run(prog)

    return result.samples

def measure_squeezed(r, phi, homodyne_angle):
    prog = sf.Program(1)

    with prog.context as q:
        Vac | q[0]
        Sgate(r, phi) | q[0] 
        MeasureHomodyne(homodyne_angle) | q[0]

    #run the engine and get the state
    eng = sf.Engine("gaussian")

    result = eng.run(prog)

    return result.samples

def measure_displaced_squeezed(s_r, s_phi, alpha, homodyne_angle):
    prog = sf.Program(1)

    d_r = np.abs(alpha)
    d_phi = np.angle(alpha)

    with prog.context as q:
        Vac | q[0]
        Sgate(s_r, s_phi) | q[0] 
        Dgate(d_r, d_phi) | q[0]
        MeasureHomodyne(homodyne_angle) | q[0]



    #run the engine and get the state
    eng = sf.Engine("gaussian")
    result = eng.run(prog)

    return result.samples