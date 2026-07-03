import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def plot_fock(n: int) -> None:

    if n > 50:
        raise ValueError("n must be less than 50")

    # --- Prepare Fock state ---
    prog_fock = sf.Program(1)
    with prog_fock.context as q:
        Fock(n) | q[0]

    eng_fock = sf.Engine("fock", backend_options={"cutoff_dim": 50})
    state_fock = eng_fock.run(prog_fock).state

    # --- Phase space grid ---
    x = np.arange(-6, 6, 0.1)
    p = np.arange(-6, 6, 0.1)
    X, P = np.meshgrid(x, p)
    Z = state_fock.wigner(0, x, p)

    fig = go.Figure(data=[go.Surface(x=X, y=P, z=Z, colorscale="Oranges", name=f'center (0,0)', showlegend=True)])

    fig.update_layout(
        title=f"State |n⟩, n = {n}",
        scene=dict(xaxis_title="<X>", yaxis_title="<P>"),
        legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
        width=600,
        height=600
        )
    
    fig.show()
        
def plot_coherent(alpha:complex) -> None:

    prog_coherent = sf.Program(1)
    re_a = np.real(alpha)
    im_a = np.imag(alpha)

    with prog_coherent.context as q:
        r = np.abs(alpha)
        phi = np.angle(alpha)
        Coherent(r, phi) | q[0]

    #run the engine and get the state
    eng_coherent = sf.Engine("gaussian")
    state_coherent = eng_coherent.run(prog_coherent).state

    #plotting
    x0 = 2*re_a
    p0 = 2*im_a
    L = 6
    x = np.linspace(x0-L, x0+L, 200)
    p = np.linspace(p0-L, p0+L, 200)
    Z = state_coherent.wigner(0, x, p)
    X, P = np.meshgrid(x, p)

    fig = go.Figure(data=[go.Surface(x=X, y=P, z=Z, colorscale="Greens", name=f"center ({2*re_a},{2*im_a})", showlegend=True)])

    fig.update_layout(
        title=f"State |a⟩; a = {re_a} + i{im_a}",
        scene=dict(xaxis_title="<X>", yaxis_title="<P>"),
        legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
        width=600,
        height=600
        )
        
    fig.show()


def plot_squeezed(r:float, phi:float) -> None:

    prog_squeezed = sf.Program(1)
    with prog_squeezed.context as q:
        Vac | q[0]
        Sgate(r, phi) | q[0] 

    #run the engine and get the state
    eng_squeezed = sf.Engine("gaussian")
    state_squeezed = eng_squeezed.run(prog_squeezed).state

    # --- Phase space grid ---
    x = np.arange(-6, 6, 0.1)
    p = np.arange(-6, 6, 0.1)
    X, P = np.meshgrid(x, p)
    Z = state_squeezed.wigner(0, x, p)

    fig = go.Figure(data=[go.Surface(x=X, y=P, z=Z, colorscale="Blues", name=f'center (0,0)', showlegend=True)])

    fig.update_layout(
        title=f"State |z⟩; z = |{r}|exp(i{phi/np.pi}π)",
        scene=dict(xaxis_title="<X>", yaxis_title="<P>"),
        legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
        width=600,
        height=600
        )
    
    fig.show()


def plot_displaced_squeezed(alpha:complex, r:float, phi:float) -> None:

    prog_dss = sf.Program(1)
    re_a = np.real(alpha)
    im_a = np.imag(alpha)

    with prog_dss.context as q:
        r_d = np.abs(alpha)
        phi_d = np.angle(alpha)
        DisplacedSqueezed(r_d, phi_d, r, phi) | q[0]

    #run the engine and get the state
    eng_dss = sf.Engine("gaussian")
    state_dss = eng_dss.run(prog_dss).state

    #plotting
    x0 = 2*re_a
    p0 = 2*im_a
    L = 6
    x = np.linspace(x0-L, x0+L, 200)
    p = np.linspace(p0-L, p0+L, 200)
    Z = state_dss.wigner(0, x, p)
    X, P = np.meshgrid(x, p)

    fig = go.Figure(data=[go.Surface(x=X, y=P, z=Z, colorscale="Purples", name=f"center ({2*re_a},{2*im_a})", showlegend=True)])

    fig.update_layout(
        title=f"State |z,a⟩; a = {re_a} + i{im_a}, z = |{r}|exp(i{phi/np.pi}π) ",
        scene=dict(xaxis_title="<X>", yaxis_title="<P>"),
        legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
        width=600,
        height=600
        )
        
    fig.show()
