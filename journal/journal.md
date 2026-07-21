# Quantum Communications Internship @ IRIDA

## Project Description


The main goal of this project is to investigate whether quantum properties, such as squeezing, can improve realistic communication channels based on coherent optical states.

We will study the phase-shift-keyed (PSK) binary communication protocol, where information is encoded and transmitted in coherent optical states and retrieved at the receiver using homodyne detection. Our aim is to determine whether the introduction of squeezing can improve the discrimination between the transmitted states and identify the optimal level of squeezing which minimizes the error probability. We will carry out this investigation both in ideal (noise-free) and realistic (noisy) channels. Our results will be directly compared with the ultimate quantum limits to discrimination, namely the Helstrom bound.

In this project we will follow closely the research paper [“Squeezing-enhanced phase-shift-keyed binary communication in noisy channels”](https://arxiv.org/abs/1710.09577) and try to reproduce their analytical results using the Strawberry Fields quantum photonic simulation framework. This implementation will provide an independent numerical validation of the reported results and enable systematic exploration of the protocol under different settings.

## Week 1 Overview (July 1–8) 

***
- Defined Objectives/Plan for project
- Read reference paper & studied theoretical background  
- Familiarized with Strawberry Fields (SF) software  
- Prepared helper functions for state visualisation and measurement in SF  
- Implemented coherent-state (CS) and displaced-squeezed-state (DSS)  communication protocols for a noise-free channel
- Derived analytically the expected probability error  
- Visualised the error probability for the aforementioned protocols and fitted it to the derived probability error

***

## Progress

***

### Read reference paper & studied theoretical background 

**Reference paper**: Gives the general idea on how the protocol works and provides the analytical expressions which we will try to reproduce using the SF software for all the different cases.

**Theoretical background**: Studied homodyne detection, squeezed/coherent states, and Helstrom bound to understand the underlying physics of the project. 

***

### Familiarized with Strawberry Fields (SF) software  

Read the [SF software documentation](https://strawberryfields.readthedocs.io/en/latest/) and spent some time to familiarize with its conventions and learn about the potential applications. 

***

### Prepared helper functions for state visualisation and measurement in SF

**State Visualisation:** Created functions that can visualise the Wigner function of a coherent state, squeezed-vacuum or displaced-squeezed vacuum according to the user's inputs. The output is just a plot.

**State Measurement:** Created functions that can perform homodyne detection for a coherent state, squeezed-vacuum or displaced-squeezed vacuum according to the user's inputs. The output is a single measurement following the statistics of the state.

***
### Implemented coherent-state (CS) and displaced-squeezed-state (DSS)    communication protocols for a noise-free channel

Homodyne measurements are performed for the exchange of coherent states, which are either prepared as $|\alpha\rangle$ or $|-\alpha\rangle$ with equal probability. For each value of $\alpha$, the measurement process is sampled many times, and the number of incorrectly classified states is recorded to estimate the error probability. This procedure is repeated over a range of values of $\alpha$ to investigate the behaviour of the error probability as a function of the average photon number.

The same protocol is then extended to displaced-squeezed states. In this case, the total energy is kept fixed while a fraction of the energy is allocated to squeezing. The error probability is studied as a function of the total photon number,

$$N_{\mathrm{tot}} = |\alpha|^2 + \sinh^2(r)$$

and the squeezing fraction,

$$\beta = \frac{\sinh^2(r)}{N_{\mathrm{tot}}}.$$

***

### Derived analytically the expected probability error

Although the reference paper outlines the initial steps of the derivation, it does not provide an explicit closed-form expression for the general error probability, but only its asymptotic behaviour in the limit $N>>1$ (Eqs. 16 & 17). Therefore, the next step was to derive the formula analytically and to check that the aforementioned limit agrees with the obtained formula:

$$P_{err} = \frac{1}{2}\text{erfc}(\frac{2 \sqrt{N_{tot}(1-\beta)}}{\sqrt2\Sigma})$$

where

$$\Sigma = (\sqrt{N_{tot}\beta}+ \sqrt{N_{tot}\beta+1})^{-1}$$

The derived expression was found to be in full agreement with the asymptotic results presented in the reference paper.

***

### Visualised the error probability for the aforementioned protocols and fitted it to the derived probability error

The produced data seem to follow well the derived error probability for both protocols and all parameters deviate less than $0.6\sigma$ from the expected value, validating our procedure so far.

***

## Next steps

- Compare the obtained results with those reported in the reference paper
- Determine the squeezing threshold for which the DSS protocol outperforms the CS protocol $(\beta_{th})$
- Determine the optimal squeezing fraction $(\beta_{opt})$
- Compare the error probability of both protocols with the corresponding Helstrom bound

## Week 2 Overview (July 8–15) 

***
- Completed noise-free protocol:
    - Compared the obtained error probability plots with those in the reference paper
    - Derived analytical expressions for $\beta_{th}$ and $\beta_{opt}$ from the reference paper
    - Determined the squeezing threshold $(\beta_{th})$ from the simulation
    - Determined the optimal squeezing fraction $(\beta_{opt})$ from the simulation
    - Compared the error probability of coherent-state (CS) protocol and displaced-squeezed state (DSS) protocol with the corresponding Helstrom bound
    - Organized all files
    
- Started phase-diffusion protocol:
    - Explored theoretical results of the paper 
    - Implemented phase-diffusion protocol in SF software and produced data for different $\sigma$ 


***

## Progress - Noise-free protocol

***

### Compared the obtained error probability plots with those in the reference paper

The simulated error probability surfaces were compared with those presented in the reference paper (Fig. 1) and were in excellent agreement over the investigated region.

***

### Derived analytical expressions for $\beta_{th}$ and $\beta_{opt}$ from the reference paper

Derived the squeezing fraction threshold $(\beta_{th})$ and optimal squeezing fraction $(\beta_{opt})$ (Eq. 8 from the paper) analytically. The squeezing threshold sets the upper limit of squeezing  $0<\beta<\beta_{th}$ for which the DSS protocol outperforms the CS protocol. The optimal squeezing fraction corresponds to the squeezing $\beta_{opt}$ that can achieve the lowest error probability. The expression for $\beta_{th}$ is obtained by setting $P^{CS}_{err}=P^{DSS}_{err}$ and $\beta_{opt}$ is found by taking $\frac{\partial P^{DSS}_{err}}{\partial \beta}|_{N, \beta_{opt}}=0$.

***

### Determined the squeezing threshold $(\beta_{th})$ from the simulation

$\beta_{th}(N)$ was determined from the intersection of the fitted error probability surfaces of the CS and DSS protocols. For $0<\beta<\beta_{th}$ the DSS protocol achieves a lower error probability than the CS protocol and is therefore advantageous. The intersection points were fitted against Eq. 8 of the paper, and the estimated parameters deviate less than $1.7\sigma$ from the theoretical predictions. 

***

### Determined the optimal squeezing fraction $(\beta_{opt})$ from the simulation

$\beta_{opt}(N)$ was determined by locating the minimum of the fitted DSS error probability surface for every average photon number $N$. The points were fitted against Eq. 8 of the paper, and the estimated parameters deviate less than $0.9\sigma$ from theoretical predictions. 

***

### Compared the error probability of CS and DSS protocol with the corresponding Helstrom bound

Visualised Helstrom bound for CS and DSS protocol (Eq. 5 of paper), and compared it against the corresponding error probability ($\beta=\beta_{opt}$ was used in the DSS protocol). For the optimal fraction of squeezing, the DSS protocol outperfoms the CS protocol when it comes to reaching the ultimate (Helstrom) bound of discrimination.
This indicates that allocating an optimal fraction of the available energy to squeezing, rather than entirely to displacement, improves the discrimination performance.

***

###  Organized all files


Arriving at the end of the noise-free protocol signified a good point to organize the existent files and notebooks to make them readable and consistent (e.g. variable names). This establishes a solid workflow which can be followed in the next, more complex protocols such that they can be navigated smoothly.

## Progress - Phase-diffusion protocol

***

### Explored theoretical results of the paper 

The next protocol that needs to be investigated is discrimination in the presence of phase noise. To implement this protocol, the theoretical framework of the paper was first studied (Eqs. 18-22). In this protocol the state becomes a statistical mixture of phase rotated CS or DSS. In particular, the state of the noise-free protocol |±a, r⟩ becomes $U_{\phi}$ |±a, r⟩ = |±a $e^{-i\phi}$, r  $e^{-i2\phi}$⟩,  where the phase $\phi$ is sampled from a Gaussian distribution centered at zero with standard deviation $\sigma$.

***

### Implemented phase-diffusion protocol in SF and produced data for different sigmas 

To model phase diffusion, a large number of the aforementioned states are generated, homodyne measurement is performed, and the measurements are averaged over all sampled phases. This procedure is repeated for different values of $\sigma$ to investigate how increasing phase noise affects the error probability.

***

## Week 3 Overview (July 15-22) 


- Fitted theoretical predictions of the error probability $P_{err}$ on data for different values of the noise parameter $\sigma$ in phase-diffused CS/DSS protocol 
- Compared the obtained error probability plots with those in the reference paper
- Determined the squeezing threshold $(\beta_{th})$ from fitted data for different values of $\sigma$
- Determined $(\beta_{th})$ numerically through theoretical expressions
- Determined the optimal squeezing fraction $(\beta_{opt})$ from fitted data for different values of $\sigma$


***

### Fitted theoretical predictions of $P_{err}$ on data for different values of the noise parameter $\sigma$

The theoretical predictions of $P_{err}$ have a non analytical form in contrary to the noise-free model, due to integrating over the phase $\phi$ (Eq. 21 of the paper). 

This expression is evaluated numerically using Gauss-Hermite quadrature for each ($N$, $\beta$) pair, and the estimated parameters are obtained through a fit which minimizes the sum of squared differences between the theoretical predictions and all simulated data points.

***

### Compared the obtained error probability plots with those in the reference paper

The simulated error probability surfaces were compared with those presented in the reference paper (Fig. 2) and showed excellent agreement over the investigated regions. As $\sigma$ grows from zero, the corresponding surfaces of CS/DSS error probabilities are shifted upwards, and the minimum of the DSS surface is shifted towards lower $\beta$ values. This observation indicates that for a fixed $N$, lower amount of squeezing is required to achieve the minimum error probability when $\sigma$ is increased.

***

### Determined the squeezing threshold $(\beta_{th})$ from fitted data for different values of $\sigma$

From the intersection of the fitted error probability surfaces of the CS and DSS protocols, $\beta_{th}$ is determined. The curve of $\beta_{th}(N)$ shifts to lower values for larger $\sigma$, indicating that noisier channels reduce the range of squeezing for which the DSS protocol can outperform the CS protocol.

***

### Found $\beta_{th}$ numerically through theoretical expressions

To  theoretically validate the $\beta_{th}$ points taken from the intersection of the surfaces, the equation $P^{DSS}_{err}-P^{CS}_{err}=0$ was solved numerically using the estimated parameters found in the previous steps. The roots were found using [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) provided from the `SciPy` library. The resulting threshold values were found to be in excellent agreement with those extracted directly from the intersection of the fitted surfaces.

***

### Determined the optimal squeezing fraction $(\beta_{opt})$ from fitted data for different values of $\sigma$

From the DSS fitted error probability surface, the minimum value for each average photon number was extracted to determine $\beta_{opt}(N)$ for different values of $\sigma$. The results verify the aforementioned expectations.

***

## Next steps:
- Investigate Helstrom bound in CS/DSS phase diffusion protocol
- Compare CS/DSS Helstrom bound to the corresponding error probability 
- Investigate $\beta_{th}$ as a function of $\sigma$ for constant values of N 
- Complete phase diffusion protocol by organizing files and ensuring smooth workflow