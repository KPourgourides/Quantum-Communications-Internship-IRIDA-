import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *


def measure_fock(n:int) -> int:

    prog = sf.Program(1)
    with prog.context as q:
        Fock(n) | q[0]
        MeasureFock() | q[0]

    #run the engine and get the state
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 100})
    result = eng.run(prog)

    return result.samples[0, 0]


def measure_coherent(alpha:complex, homodyne_angle:float) -> float:

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

def measure_squeezed(r:float, phi:float, homodyne_angle:float) -> float:

    prog = sf.Program(1)
    with prog.context as q:
        Vac | q[0]
        Sgate(r, phi) | q[0] 
        MeasureHomodyne(homodyne_angle) | q[0]

    #run the engine and get the state
    eng = sf.Engine("gaussian")

    result = eng.run(prog)

    return result.samples[0, 0]

def measure_displaced_squeezed(r_s:float, phi_s:float, alpha:complex, homodyne_angle:float):

    prog = sf.Program(1)
    r_d = np.abs(alpha)
    phi_d = np.angle(alpha)

    with prog.context as q:
        Vac | q[0]
        Sgate(r_s, phi_s) | q[0] 
        Dgate(r_d, phi_d) | q[0]
        MeasureHomodyne(homodyne_angle) | q[0]

    #run the engine and get the state
    eng = sf.Engine("gaussian")
    result = eng.run(prog)

    return result.samples[0, 0]