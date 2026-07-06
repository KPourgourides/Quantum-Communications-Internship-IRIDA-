# Quantum Communications Internship @ IRIDA

## Week 1 (July 1–8)

## Milestones
- Define Objectives/Plan for project  
- Read reference paper & study theoretical background (homodyne detection, squeezed/coherent states, Helstrom bound)  
- Familiarize with Strawberry Fields (SF) software  
- Prepare helper functions for state visualisation and measurement in SF  
- Prepare protocols for coherent functions and displaced squeezed functions in noise-free channel  
- Visualise the error probability for the aforementioned protocols and compare to theoretical function

***

## Progress

### Define Objectives for Project

The main goal of this project is to investigate whether quantum properties, such as squeezing, can improve realistic communication channels based on coherent optical states.

We will study the phase-shift-keyed (PSK) binary communication protocol, where information is encoded and transmitted in coherent optical states and retrieved at the receiver using homodyne detection. Our aim is to determine whether the introduction of squeezing can improve the discrimination between the transmitted states and identify the optimal level of squeezing which minimizes the error probability. We will carry out this investigation both in ideal (noise-free) and realistic (noisy) channels. Our results will be directly compared with the ultimate quantum limits to discrimination, namely the Helstrom bound.

In this project we will follow closely the research paper [“Squeezing-enhanced phase-shift-keyed binary communication in noisy channels”](https://arxiv.org/abs/1710.09577) and try to reproduce their analytical results using the Strawberry Fields quantum photonic simulation framework. This implementation will provide an independent numerical validation of the reported results and enable systematic exploration of the protocol under different settings.

**Theoretical Objectives**
- Understand the principles of PSK binary communication  
- Investigate the effect of squeezing on quadrature statistics and state discrimination  
- Analyse the effect of channel imperfections on the communication performance  
- Study Helstrom bound and compare with homodyne-based detection  

**Computational Objectives**
- Become proficient in the Strawberry Fields quantum photonic simulation framework  
- Simulate Gaussian states in various channels and perform homodyne detection  
- Investigate the influence of key system parameters (e.g. squeezing) on performance  
- Validate the simulated results against the theoretical predictions of the paper  

***

### Plan to Achieve the Objectives

The theoretical and computational objectives will be developed in parallel so that the understanding of the underlying physics continuously informs the implementation, while simulation results provide numerical validation of the theoretical predictions.

- Familiarization with Strawberry Fields  
  Develop an intuitive understanding on how the Strawberry Fields framework works by studying its capabilities (e.g. state and channel preparation, measurement operations).

- Development and Validation of Core Simulation Components  
  Implement a collection of helper functions performing specific tasks and validate them independently against theoretical results before integrating it into the complete protocol.

- Progressive Construction of the Communication Model  
  Combine the validated components to build a variety of communication scenarios. This approach facilitates debugging, validation, and future extensions of the simulation.

- Performance Evaluation and Comparison  
  Compare error probabilities with the Helstrom bound to assess the potential advantages of squeezing-enhanced communication under different conditions.

- Theoretical Analysis and Verification  
  Derive the analytical expressions relevant to the project and compare simulated results with the analytical predictions of the reference paper.

***
### Prepare helper functions for state visualisation and measurement in SF

**State Visualisation:** Created functions that can visualise the Wigner function of a coherent state, squeezed-vacuum or displaced-squeezed vacuum according to user's inputs. The output is just a plot.

**State Measurement:** Created functions that can perform homodyne detection for a coherent state, squeezed-vacuum or displaced-squeezed vaccum according to user's inputs. The output is a single measurement following the statistics of the state.

***
### Prepare protocols for coherent functions and displaced squeezed functions in noise-free channel  

Homodyne measurements are performed for the exchange of coherent states, which are either prepared as $|\alpha\rangle$ or $|-\alpha\rangle$ with equal probability. For each value of $\alpha$, the measurement process is sampled many times, and the number of incorrectly classified states is recorded to estimate the error probability. This procedure is repeated over a range of values of $\alpha$ to investigate the behaviour of the error probability as a function of the average photon number.

The same protocol is then extended to displaced squeezed states. In this case, the total energy is kept fixed while a fraction of the energy is allocated to squeezing. The error probability is studied as a function of the total photon number,

$$N_{\mathrm{tot}} = |\alpha|^2 + \sinh^2(r)$$

and the squeezing fraction,

$$\beta = \frac{\sinh^2(r)}{N_{\mathrm{tot}}}.$$

