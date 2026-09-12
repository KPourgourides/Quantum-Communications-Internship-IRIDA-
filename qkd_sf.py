import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import*
import plotly.graph_objects as go


#=========================================================================================================
#                        PROTOCOL SF
#=========================================================================================================

def sf_protocol(eta, var_a, keylen, V):

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

            alpha_p = np.random.normal(0, np.sqrt(var_a))
            alpha_q = np.random.normal(0, np.sqrt(var_a))
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
#                        PROTOCOL 
#=========================================================================================================

def protocol(eta, var_a, keylen, V):

    alpha_q = np.random.normal(loc=0, scale=np.sqrt(var_a), size=keylen)
    alpha_p = np.random.normal(loc=0, scale=np.sqrt(var_a), size=keylen)
    base = np.random.randint(0, 2, size=keylen)
    bob_bitstr = np.zeros(keylen)
    alice_bitstr = np.zeros(keylen)
    
    noise_q = np.sqrt(eta*V+(1-eta))
    noise_p = np.sqrt(eta/V+(1-eta))

    for i, choice in enumerate(base):

        if choice == 0:  

            bob_bitstr[i] = np.random.normal(loc=np.sqrt(eta)*alpha_q[i], scale=noise_q)
            alice_bitstr[i] = alpha_q[i]

        elif choice == 1:   

            bob_bitstr[i] = np.random.normal(loc=np.sqrt(eta)*alpha_p[i], scale=noise_p)
            alice_bitstr[i] = alpha_p[i]

    return alice_bitstr, bob_bitstr, base

#=========================================================================================================
#                        MUTUAL INFORMATION
#=========================================================================================================

def theoretical_I_AB(var_a, eta, V):

    I_AB = 0.5*(1/2*np.log2(1+(var_a*eta)/(1+eta*(V-1))) +  1/2*np.log2(1+(var_a*eta)/(1+eta*(1/V-1))))
    I_AB_x = 1/2*np.log2(1+(var_a*eta)/(1+eta*(V-1))) 
    I_AB_p = 1/2*np.log2(1+(var_a*eta)/(1+eta*(1/V-1))) 
    return I_AB, I_AB_x, I_AB_p


def bitstr_to_base(bitstr, base, quadr):

    return [val for i, val in enumerate(bitstr) if base[i] == quadr]


def MC_I_AB(var_a_grid, eta, keylen, V):

    size = len(var_a_grid)
    I_grid = np.zeros((size))
    I_grid_x = np.zeros((size))
    I_grid_p = np.zeros((size))

    for i,var_a in enumerate(var_a_grid):
        alice_bitstr, bob_bitstr, base = protocol(eta, var_a, keylen, V)
        alice_bitstr_x = bitstr_to_base(alice_bitstr, base, 0)
        bob_bitstr_x = bitstr_to_base(bob_bitstr, base, 0)

        alice_bitstr_p = bitstr_to_base(alice_bitstr, base, 1)
        bob_bitstr_p = bitstr_to_base(bob_bitstr, base, 1)

        rho_x = np.corrcoef(alice_bitstr_x, bob_bitstr_x)[0, 1]
        I_AB_x = -0.5 * np.log2(1 - rho_x**2)

        rho_p = np.corrcoef(alice_bitstr_p, bob_bitstr_p)[0, 1]
        I_AB_p = -0.5 * np.log2(1 - rho_p**2)
        
        I_grid[i] = 0.5*(I_AB_x+ I_AB_p)
        I_grid_x[i] = I_AB_x
        I_grid_p[i] = I_AB_p
        
    return I_grid, I_grid_x, I_grid_p


def get_mutual_information_MC(var_a_grid, V_grid, eta_grid, keylen):
    Ix_per_eta_dict_MC = {}

    for j, eta in enumerate(eta_grid):
        I_grid_x = np.zeros((len(var_a_grid), len(V_grid)))
        for i,v in enumerate(V_grid):
            I, I_x, I_p = MC_I_AB(var_a_grid, eta, keylen, v)
            I_grid_x[:,i] = I_x
            print(f"\rProgress: {i+1}/{len(V_grid)}", end="", flush=True)
        Ix_per_eta_dict_MC[f'eta_{eta}'] = I_grid_x
        print(f"\rProgress: {j+1}/{len(eta_grid)}", end="", flush=True)
    return Ix_per_eta_dict_MC


def get_mutual_information_th(var_a_grid, V_grid, eta_grid):
    Ix_per_eta_dict_th = {}

    for j, eta in enumerate(eta_grid):
        th_I_grid_x = np.zeros((len(var_a_grid), len(V_grid)))

        for i,v in enumerate(V_grid):
            th_I, th_I_x, th_I_p = theoretical_I_AB(var_a_grid, eta, v)
            th_I_grid_x[:,i] = th_I_x

        Ix_per_eta_dict_th[f'eta_{eta}'] = th_I_grid_x
        print(f"\rProgress: {j+1}/{len(eta_grid)}", end="", flush=True)
    return Ix_per_eta_dict_th


#=========================================================================================================
#                        KEY RATE
#=========================================================================================================

def key_rate(b, I, chi):
    return b*I-chi

#=========================================================================================================
#                             HOLEVO BOUND 
#=========================================================================================================

def G(x):
    if x <= 1:
        return 0.0

    n = (x - 1) / 2

    return (n + 1)*np.log2(n + 1) - n*np.log2(n)


def get_Holevo_bound(var_a_grid, V_grid, eta_grid, protocol):
    
    chix_per_eta_dict = {}
    
    for k,eta in enumerate(eta_grid):
        th_chix_grid = np.zeros((len(var_a_grid), len(V_grid)))
        for i,var_a in enumerate(var_a_grid):
            for j,v in enumerate(V_grid):
                if protocol=='RR':
                    th_chix_grid[i][j] = holevo_RR_SF(eta, var_a, v, 'x')
                elif protocol=='DR':
                    th_chix_grid[i][j] = holevo_DR_SF(eta, var_a, v, 'x')
        chix_per_eta_dict[f'eta_{eta}'] = th_chix_grid
        print(f"\rProgress: {k+1}/{len(eta_grid)}", end="", flush=True)

    return chix_per_eta_dict



#=========================================================================================================
#                        HOLEVO BOUND ANALYTICAL
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


def holevo_DR(var_a, eta, V):

    # Alice's squeezed state + classical modulation
    sx = V + var_a
    sp = 1/V + var_a

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
    ep_cond_p = (1 - eta)/V + eta
    lambda_cond_p = np.sqrt(ex_cond_p * ep_cond_p)
    chi_p = G(lambda_E) - G(lambda_cond_p)

    return chi_x, chi_p


# =========================================================================================================
#                         HOLEVO BOUND SF
# =========================================================================================================

def sf_covariance(eta, V):

    r = -np.log(V)/2
    theta = np.arccos(np.sqrt(eta))
    prog = sf.Program(2)

    with prog.context as q:

        Vac | q[0]
        Sgate(r, 0) | q[0]
        Vac | q[1]
        BSgate(theta, 0.0) | (q[0], q[1])

    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    gamma_BE = result.state.cov()

    return gamma_BE


def holevo_DR_SF(eta, var_a, V, basis="x"):

    # Quantum covariance
    gamma_BE = sf_covariance(eta, V)
    gamma_E_q = np.array([[gamma_BE[1][1], gamma_BE[1][3]], [gamma_BE[3][1], gamma_BE[3][3]]])

    # Adding classical variance
    gamma_E = gamma_E_q + (1 - eta) * var_a * np.eye(2)

    # Conditional covariance on Alice's result
    gamma_E_cond = gamma_E.copy()

    if basis == "x":
        gamma_E_cond[0, 0] -=  (1 - eta) * var_a

    elif basis == "p":
        gamma_E_cond[1, 1] -=  (1 - eta) * var_a

    nu_E = np.sqrt(np.linalg.det(gamma_E))
    nu_cond = np.sqrt(np.linalg.det(gamma_E_cond))

    return G(nu_E) - G(nu_cond)


def holevo_RR_SF(eta, var_a, V, basis="x"):

    # Quantum covariance from SF
    gamma_BE_q = sf_covariance(eta, V)

    # Adding classical variance
    s = np.sqrt(eta * (1 - eta))

    gamma_mod = var_a * np.array([
        [eta, s, 0, 0],
        [s, 1 - eta, 0, 0],
        [0, 0, eta, s],
        [0, 0, s, 1 - eta]])

    gamma_BE = gamma_BE_q + gamma_mod

    # Eve's covariance
    gamma_E = gamma_BE[np.ix_([1, 3], [1, 3])]

    # Condition on Bob's measurement
    if basis == "x":

        V_B = gamma_BE[0, 0]
        c = gamma_BE[[1, 3], 0]

    elif basis == "p":

        V_B = gamma_BE[2, 2]
        c = gamma_BE[[1, 3], 2]

    gamma_E_cond = gamma_E - np.outer(c, c) / V_B

    nu_E = np.sqrt(np.linalg.det(gamma_E))
    nu_cond = np.sqrt(np.linalg.det(gamma_E_cond))

    return G(nu_E) - G(nu_cond)

# =========================================================================================================
#                         SQUEEZING THRESHOLD
# =========================================================================================================

def cs_b_th(I_per_eta_dict_th, chix_per_eta_dict, eta_grid):

    bs = np.linspace(0.0, 1, 2000)
    low_b_cs=np.ones((len(eta_grid)))

    for i,eta_DR in enumerate(eta_grid):
        th_I_grid_x = I_per_eta_dict_th[f'eta_{eta_DR}'][:, -1]
        th_chi_x_grid = chix_per_eta_dict[f'eta_{eta_DR}'][:, -1]
        for b in bs:
            K_grid = key_rate(b, th_I_grid_x, th_chi_x_grid)
            max_value = np.nanmax(K_grid)
            if max_value>=1e-3:
                low_b_cs[i]=b
                break
    return low_b_cs


def b_th(I_per_eta_dict_th, chix_per_eta_dict, eta_grid):

    bs = np.linspace(0.0, 1, 2000)
    low_b = np.ones((len(eta_grid)))

    for i,eta_DR in enumerate(eta_grid):
        th_I_grid_x = I_per_eta_dict_th[f'eta_{eta_DR}'][:, :-1]
        th_chi_x_grid = chix_per_eta_dict[f'eta_{eta_DR}'][:, :-1]
        for b in bs:
            K_grid = key_rate(b, th_I_grid_x, th_chi_x_grid)
            max_value = np.nanmax(K_grid)
            if max_value>=1e-3:
                low_b[i]=b
                break
    return low_b

# =========================================================================================================
#                        OPTIMUM
# =========================================================================================================

def var_a_optimal_data(I, chi, bs, var_a) -> dict:

    var_a_opt_dict = {}

    for b in bs:
        K = key_rate(b, I, chi)
        idx = np.nanargmax(K, axis=0)
        var_a_opt = var_a[idx]
        var_a_opt_dict[f'b_{b}'] = var_a_opt

    return var_a_opt_dict

# =========================================================================================================
#                         PLOTS
# =========================================================================================================

def plot_I_chi(eta,  V_grid, var_a_grid, I_per_eta_dict_th, chix_per_eta_dict, I_per_eta_dict_MC, MC = 'False'):

    th_I_grid_x_RR = I_per_eta_dict_th[f'eta_{eta}']
    th_chi_x_grid = chix_per_eta_dict[f'eta_{eta}']

    if MC:
        print('hi')
        I_grid_x = I_per_eta_dict_MC[f'eta_{eta}']

    fig = go.Figure()
    V_mesh, var_a_mesh = np.meshgrid(V_grid, var_a_grid)

    if MC:
        fig.add_trace(go.Scatter3d(x=V_mesh.ravel(), y=var_a_mesh.ravel(), z=I_grid_x.ravel(), mode='markers', marker=dict(size=3, color='blue')))
    fig.add_trace(go.Surface(x=V_grid, y=var_a_grid, z=th_I_grid_x_RR, surfacecolor=np.zeros_like(th_I_grid_x_RR), colorscale=[[0.0, 'magenta'], [1.0, 'blue']], showscale=False))
    fig.add_trace(go.Surface(x=V_grid, y=var_a_grid, z=th_chi_x_grid, surfacecolor=np.zeros_like(th_chi_x_grid), colorscale=[[0.0, 'red'], [1.0, 'red']], showscale=False))
    fig.update_layout(scene=dict(xaxis_title="V", yaxis_title=r"var_a", zaxis = dict(title="Info"), aspectmode ="cube"), width=900, height=750)
    fig.show()


def plot_K(b, eta, var_a_grid, V_grid, I_per_eta_dict_th, I_per_eta_dict_MC, chix_per_eta_dict, MC=False):

    th_I_grid_x = I_per_eta_dict_th[f'eta_{eta}']
    if MC:
        I_grid_x = I_per_eta_dict_MC[f'eta_{eta}']
        K = key_rate(b, I_grid_x, th_chi_x_grid)
    th_chi_x_grid = chix_per_eta_dict[f'eta_{eta}']

    th_K = key_rate(b, th_I_grid_x, th_chi_x_grid)
    

    fig = go.Figure()
    V_mesh, var_a_mesh = np.meshgrid(V_grid, var_a_grid)
    fig.add_trace(go.Surface(x=V_grid, y=var_a_grid, z= th_K))
    fig.add_trace(go.Surface(x=V_grid, y=var_a_grid, z=np.zeros((len(var_a_grid), len(V_grid)))))
    if MC:
        fig.add_trace(go.Scatter3d(x=V_mesh.ravel(), y=var_a_mesh.ravel(), z=K.ravel(), mode='markers', marker=dict(size=3, color='blue')))
    fig.update_layout(scene=dict(xaxis_title="V", yaxis_title="var_a", zaxis = dict(title="K"), aspectmode ="cube"), width=900, height=750)
    fig.show()


def plot_2D_chi_I_K(b, eta, V, var_a_grid, V_grid, I_per_eta_dict_th, I_per_eta_dict_MC, chix_per_eta_dict, MC=False):

    th_I_grid_x = I_per_eta_dict_th[f'eta_{eta}']
    I_grid_x = I_per_eta_dict_MC[f'eta_{eta}']
    th_chi_x_grid = chix_per_eta_dict[f'eta_{eta}']

    idx = np.where(V_grid==V)[0][0]
    th_I_x = (th_I_grid_x[:, idx])
    th_chi_x = (th_chi_x_grid[:, idx])
    
    th_K = key_rate(b, th_I_x, th_chi_x)

    if MC:
        I_grid_x = (I_grid_x[:, idx])
        K = key_rate(b, I_grid_x, th_chi_x)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), dpi=100)

    ax[0].set_title(rf'$\beta$={b}, V={V} and $\eta$={eta}', fontsize=15)
    if MC:
        ax[0].scatter(var_a_grid, I_grid_x, s=15, color = 'magenta')
    ax[0].plot(var_a_grid, th_I_x, '--', color='magenta', label=r'$I_{AB}$')
    ax[0].plot(var_a_grid, th_chi_x, 'b--', label=r'$\chi_{BE}$')
    ax[0].set_ylabel(r'Information', fontsize=15)
    ax[0].legend(fontsize=15)

    ax[1].plot(var_a_grid, th_K, '--r', label = f'Key-Rate')
    if MC:
        ax[1].scatter(var_a_grid, K, s=15, color = 'red', label = f'MC Key-Rate')
    ax[1].axhline(0)
    ax[1].set_xlabel(r'$Var_{{\alpha}}$', fontsize=15)
    ax[1].set_ylim(0)
    ax[1].set_ylabel(r'$K$', fontsize=15)
    plt.grid(True)
    plt.tight_layout()


def plot_K_regions(eta_grid, I_per_eta_dict_th, chix_per_eta_dict):

    low_b = b_th(I_per_eta_dict_th, chix_per_eta_dict, eta_grid)
    low_b_cs = cs_b_th(I_per_eta_dict_th, chix_per_eta_dict, eta_grid)

    plt.plot(eta_grid, low_b, '--', color='k', linewidth=2)
    plt.plot(eta_grid, low_b_cs, '--', color='k', linewidth=2)
    plt.fill_between(eta_grid, low_b, low_b_cs, color='gold', alpha=0.6)
    plt.fill_between(eta_grid, low_b_cs, 1, color='green', alpha=0.6)
    plt.fill_between(eta_grid, low_b, 0, color='red', alpha=0.6)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$b_{th}$')


def plot_opt(b, eta, var_a_grid, V_grid, I_per_eta_dict_th, chix_per_eta_dict):

    th_I_grid_x = I_per_eta_dict_th[f'eta_{eta}']
    th_chi_x_grid = chix_per_eta_dict[f'eta_{eta}']

    bs = np.linspace(0.90, 1, 11)
    var_a_opt_dict = var_a_optimal_data(th_I_grid_x, th_chi_x_grid, bs, var_a_grid)

    fig = plt.figure(figsize=(8, 5), dpi=150)
    for b in bs:
        plt.plot(V_grid, var_a_opt_dict[f'b_{b}'], label=f'{b:.2f}')
    plt.xlabel('V')
    plt.ylabel('var_a optimal')
    plt.legend()
