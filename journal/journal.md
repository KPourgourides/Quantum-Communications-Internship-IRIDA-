# Quantum Communications Internship @ IRIDA

## Project Description


The main goal of this project is to investigate whether quantum properties, such as squeezing, can improve realistic communication channels based on coherent optical states.

We will study the phase-shift-keyed (PSK) binary communication protocol, where information is encoded and transmitted in coherent optical states and retrieved at the receiver using homodyne detection. Our aim is to determine whether the introduction of squeezing can improve the discrimination between the transmitted states and identify the optimal level of squeezing which minimizes the error probability. We will carry out this investigation both in ideal (noise-free) and realistic (noisy) channels. Our results will be directly compared with the ultimate quantum limits to discrimination, namely the Helstrom bound.

In this project we will follow closely the research paper [“Squeezing-enhanced phase-shift-keyed binary communication in noisy channels”](https://arxiv.org/abs/1710.09577) and try to reproduce their analytical results using the Strawberry Fields quantum photonic simulation framework. This implementation will provide an independent numerical validation of the reported results and enable systematic exploration of the protocol under different settings.

## Week 1 Overview (July 1–8) 

***
- Defined Objectives/Plan for project
- Read reference paper & studied theoretical background (homodyne detection, squeezed/coherent states, Helstrom bound)  
- Familiarized with Strawberry Fields (SF) software  
- Prepared helper functions for state visualisation and measurement in SF  
- Prepared protocols for coherent states and displaced-squeezed states in noise-free channel  
- Visualised the error probability and Helstrom bound for the aforementioned protocols 

***

## Progress

***

### Prepared helper functions for state visualisation and measurement in SF

**State Visualisation:** Created functions that can visualise the Wigner function of a coherent state, squeezed-vacuum or displaced-squeezed vacuum according to user's inputs. The output is just a plot.

**State Measurement:** Created functions that can perform homodyne detection for a coherent state, squeezed-vacuum or displaced-squeezed vaccum according to user's inputs. The output is a single measurement following the statistics of the state.

***
### Prepared protocols for coherent states and displaced-squeezed states in noise-free channel  

Homodyne measurements are performed for the exchange of coherent states, which are either prepared as $|\alpha\rangle$ or $|-\alpha\rangle$ with equal probability. For each value of $\alpha$, the measurement process is sampled many times, and the number of incorrectly classified states is recorded to estimate the error probability. This procedure is repeated over a range of values of $\alpha$ to investigate the behaviour of the error probability as a function of the average photon number.

The same protocol is then extended to displaced-squeezed states. In this case, the total energy is kept fixed while a fraction of the energy is allocated to squeezing. The error probability is studied as a function of the total photon number,

$$N_{\mathrm{tot}} = |\alpha|^2 + \sinh^2(r)$$

and the squeezing fraction,

$$\beta = \frac{\sinh^2(r)}{N_{\mathrm{tot}}}.$$

