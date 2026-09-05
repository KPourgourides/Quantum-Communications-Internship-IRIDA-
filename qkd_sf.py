import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import*


#=========================================================================================================
#                        PROTOCOL
#=========================================================================================================

def sf_protocol(eta, var, key_len):

    bob_bitstr = []
    alice_bitstr = []
    for i in range(key_len):
        prog = sf.Program(2)
        theta = np.arccos(np.sqrt(eta))
        with prog.context as q:

            Vac | q[0]
            alpha_p = np.random.normal(0, np.sqrt(var))
            alpha_q = np.random.normal(0, np.sqrt(var))
            Xgate(alpha_q) | q[0]
            Zgate(alpha_p) | q[0]

            Vac | q[1]
            
            BSgate(theta, 0.0) | (q[0], q[1])

            x = np.random.randint(0, 2)
            if x==0:
                MeasureHomodyne(0) | q[0]
                alice_bitstr.append(alpha_q)
                
            else:
                MeasureHomodyne(np.pi/2) | q[0]
                alice_bitstr.append(alpha_p)

        eng = sf.Engine("gaussian")
        result = eng.run(prog)
        bob_bitstr.append(result.samples[0][0])
        eng.reset()

    return alice_bitstr, bob_bitstr

#=========================================================================================================
#                        MUTUAL INFORMATION
#=========================================================================================================

def theoretical_I_AB(var_a, eta):
    return 1/2*np.log2(1+(var_a*eta))

#=========================================================================================================
#                        KEY RATE
#=========================================================================================================

def key_rate(b, I, x):
    return b*np.array(I)-np.array(x)

#=========================================================================================================
#                        HOLEVO BOUND
#=========================================================================================================

def theoretical_holevo_bound(lambda1, lambda2):
    def G(x):

        if x==0:
            return 0
        return  (x + 1)*np.log2(x + 1) - x*np.log2(x)
    return G((lambda1 - 1)/2) - G((lambda2 - 1)/2)


def lambda_RR(var_a, eta):

    #lambda1 = 1+(1-eta)*var_a
    #lambda2 = np.sqrt((1+(1-eta)*var_a)*(1+(1-eta)*var_a/(1+eta*var_a)))

    V1 = var_a+1+np.sqrt(2*var_a+var_a**2)
    V2 = var_a+1-np.sqrt(2*var_a+var_a**2)

    VB_p = 1/(2*V1)+1/(2*V2)
    VB_x = (V1+V2)/2
    CBE_x = np.sqrt(eta*(1-eta))*(1-VB_x)

    lambda1 = np.sqrt(VB_x*(1-eta)+eta)*np.sqrt(VB_p*(1-eta)+eta)
    lambda2 =  np.sqrt((VB_x*(1-eta)+eta-CBE_x**2/(VB_x*eta+1-eta))*(VB_p*(1-eta)+eta))

    return lambda1, lambda2

def lambda_DR(var_a, eta):
    VE = 1 + (1 - eta) * var_a

    lambda1 = VE
    lambda2 = np.sqrt(VE)

    return lambda1, lambda2


