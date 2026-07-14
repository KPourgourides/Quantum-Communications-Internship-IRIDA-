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
- Implemented coherent-state (CS) and displaced-squeezed-state (DSS)  communication protocols for a noise-free channel.
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

$$\Sigma = (\sqrt{N\beta}+ \sqrt{N\beta+1})^{-1}$$

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
- Finished noise-free protocol:
    - Compared the obtained results with those reported in the reference paper
    - Derived analytically expressions from the reference paper
    - Determined the squeezing threshold for which the DSS protocol outperforms the CS protocol $(\beta_{th})$
    - Determined the optimal squeezing fraction $(\beta_{opt})$
    - Compared the error probability of both protocols with the corresponding Helstrom bound
    - Organized all files
- Started phase-diffusion protocol:
    - Explored theoretical results of the paper 
    - Implemented phase-diffusion protocol in SF
    - Produced data for different sigmas 
    - Investigated how to fit theoretical predictions on data (complicated expression)


***