import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import*


#=========================================================================================================
#                        PROTOCOL
#=========================================================================================================

def sf_protocol(eta, var, keylen, V):

    bob_bitstr = np.zeros((keylen))
    alice_bitstr = np.zeros((keylen))
    base = np.zeros((keylen))

    r = -np.log(V)/2
    for i in range(keylen):
        prog = sf.Program(2)
        theta = np.arccos(np.sqrt(eta))
        with prog.context as q:

            Vac | q[0]
            Sgate(r, 0) | q[0] 

            alpha_p = np.random.normal(0, np.sqrt(var))
            alpha_q = np.random.normal(0, np.sqrt(var))
            Xgate(alpha_q) | q[0]
            Zgate(alpha_p) | q[0]

            Vac | q[1]
            
            BSgate(theta, 0.0) | (q[0], q[1])

            x = np.random.randint(0, 2)
            if x==0:
                MeasureHomodyne(0) | q[0]
                alice_bitstr[i] = alpha_q
                
            else:
                MeasureHomodyne(np.pi/2) | q[0]
                alice_bitstr[i] = alpha_p

        base[i] = x
        eng = sf.Engine("gaussian")
        result = eng.run(prog)
        bob_bitstr[i]= result.samples[0][0]
        eng.reset()

    return alice_bitstr, bob_bitstr, base

#=========================================================================================================
#                        MUTUAL INFORMATION
#=========================================================================================================

def theoretical_I_AB(var_a, eta, V):

    I_AB = 0.5*(1/2*np.log2(1+(var_a*eta)/(1+eta*(V-1))) +  1/2*np.log2(1+(var_a*eta)/(1+eta*(1/V-1))))
    I_AB_x = 1/2*np.log2(1+(var_a*eta)/(1+eta*(V-1))) 
    I_AB_p = 1/2*np.log2(1+(var_a*eta)/(1+eta*(1/V-1))) 
    return I_AB, I_AB_x, I_AB_p


def bitstr_to_base(bitstr, quadr, base):

    return [val for i, val in enumerate(bitstr) if base[i] == quadr]


def MC_I_AB(var_a_grid, eta, keylen, V):

    size = len(var_a_grid)
    I_grid = np.zeros((size))
    I_grid_x = np.zeros((size))
    I_grid_p = np.zeros((size))

    for i,var_a in enumerate(var_a_grid):
        alice_bitstr, bob_bitstr, base = sf_protocol(eta, var_a, keylen, V)
        alice_bitstr_x = bitstr_to_base(alice_bitstr, 0, base)
        bob_bitstr_x = bitstr_to_base(bob_bitstr, 0, base)

        alice_bitstr_p = bitstr_to_base(alice_bitstr, 1, base)
        bob_bitstr_p = bitstr_to_base(bob_bitstr, 1, base)

        rho_x = np.corrcoef(alice_bitstr_x, bob_bitstr_x)[0, 1]
        I_AB_x = -0.5 * np.log2(1 - rho_x**2)

        rho_p = np.corrcoef(alice_bitstr_p, bob_bitstr_p)[0, 1]
        I_AB_p = -0.5 * np.log2(1 - rho_p**2)
        
        I_grid[i] = 0.5*(I_AB_x+ I_AB_p)
        I_grid_x[i] = I_AB_x
        I_grid_p[i] = I_AB_p

        print(f"\rProgress: {i+1}/{len(var_a_grid)}", end="", flush=True)

    return I_grid, I_grid_x, I_grid_p
    

#=========================================================================================================
#                        KEY RATE
#=========================================================================================================

def key_rate(b, I, x):
    return b*np.array(I)-np.array(x)

#=========================================================================================================
#                             HOLEVO BOUND 
#=========================================================================================================

def G(x):
    # nu is a symplectic eigenvalue, vacuum = 1
    if x <= 1 + 1e-12:
        return 0.0

    n = (x - 1) / 2

    return (n + 1)*np.log2(n + 1) - n*np.log2(n)


#=========================================================================================================
#                        HOLEVO BOUND RR PROTOCOL
#=========================================================================================================


def holevo_RR(var_a, eta, V):

    l1, l2, l3x, l3p = lambda_RR(var_a, eta, V)

    S_AB = G(l1) + G(l2)

    S_A_given_Bx = G(l3x)
    S_A_given_Bp = G(l3p)
    S_cond = 0.5*(S_A_given_Bx + S_A_given_Bp)

    chi = S_AB - S_cond
    chi_x = S_AB - S_A_given_Bx
    chi_p = S_AB - S_A_given_Bp

    return chi, chi_x, chi_p


def lambda_RR(var_a, eta, V):

    # Covariance of Alice's modulated transmitted state
    sx = V + var_a
    sp = 1/V + var_a

    # Symplectic eigenvalue of the modulated state
    mu = np.sqrt(sx * sp)

    # Purification
    cx = np.sqrt((sx/mu) * (mu**2 - 1))
    cp = np.sqrt((sp/mu) * (mu**2 - 1))

    # Bob after pure-loss channel
    bx = eta*sx + (1 - eta)
    bp = eta*sp + (1 - eta)
 
    # AB covariance matrix
    detA = sx*sp
    detB = bx * bp
    detC = -eta * cx * cp

    Delta = detA + detB + 2*detC
    detAB = ((mu*bx - eta*cx**2)*(mu*bp - eta*cp**2))
    discriminant = Delta**2 - 4*detAB

    lambda1 = np.sqrt((Delta + np.sqrt(discriminant))/2)
    lambda2 = np.sqrt((Delta - np.sqrt(discriminant))/2)
    lambda3_x = np.sqrt(mu * (mu - eta*cx**2/bx))
    lambda3_p = np.sqrt(mu * (mu - eta*cp**2/bp))

    return lambda1, lambda2, lambda3_x, lambda3_p


# =========================================================================================================
#                         HOLEVO BOUND DR PROTOCOL
# =========================================================================================================

def holevo_DR(var_a, eta, V):

    # Alice's squeezed state + classical modulation
    sx = V + var_a
    sp = 1 / V + var_a

    # Eve unconditional covariance
    ex = (1 - eta) * sx + eta
    ep = (1 - eta) * sp + eta
    lambda_E = np.sqrt(ex * ep)

    # Eve conditioned on Alice's X modulation
    ex_cond_x = (1 - eta) * V + eta
    ep_cond_x = (1 - eta) * sp + eta
    lambda_cond_x = np.sqrt(ex_cond_x * ep_cond_x)
    chi_x = G(lambda_E) - G(lambda_cond_x)
    
    # Eve conditioned on Alice's P modulation
    ex_cond_p = (1 - eta) * sx + eta
    ep_cond_p = (1 - eta) / V + eta
    lambda_cond_p = np.sqrt(ex_cond_p * ep_cond_p)
    chi_p = G(lambda_E) - G(lambda_cond_p)

    return chi_x, chi_p










