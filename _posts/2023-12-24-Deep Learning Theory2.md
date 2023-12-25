# deeplearningtheory

**Author:** Hyunin Lee  
**Date:** September 2023


## Day2
Lecture 2 is about Quadratic models and nearly-kernel methods. Chapter 11.4, chapter7.2 and chapter ∞.2.2 covers

### Linear Models and Kernel Methods

Two forms of a solution for a linear model:

- parameter space - linear regression

  $$ z_i(x_{\dot{\beta}}; \theta^*) = \sum_{j=0}^{n_f} W_{ij}^* \phi_j(x_{\dot{\beta}}) $$

- sample space - kernel methods

  $$ z_i(x_{\dot{\beta}}; \theta^*) = \sum_{\tilde{\alpha}_1, \tilde{\alpha}_2 \in A} k_{\dot{\beta} \tilde{\alpha}_1} \tilde{k}^{\tilde{\alpha}_1 \tilde{\alpha}_2} y_{i;\tilde{\alpha}_2} $$


### Nonlinear models

Let's relax the above linear model into a nonlinear model, specifically a blue{quadratic model}.

$$
z_{i;\delta}(\theta) = \sum_{j=0}^{n_f} W_{ij} \phi_j(x_\delta) + \textcolor{blue}{\frac{\epsilon}{2} \sum_{j_1, j_2 = 0}^{n_f} W_{i j_1} W_{i j_2} \psi_{j_1 j_2}(x_\delta)}
$$

- It's nonlinear because it's quadratic in the weights: $ W_{ij_1} W_{ij_2} $.
- $ \varepsilon \ll 1 $ is a small parameter that controls the size of the deformation.
- We've introduced $ \frac{(n_f + 1)(n_f + 2)}{2} $ meta feature functions, $ \psi_{j_1 j_2} (x) $, with two feature indices.

### Quadratic models

To familiarize ourselves with this model, let's make a small change in the model parameters $W_{ij} \to W_{ij} + dW_{ij}$:

$$
z_i(x_\delta; \theta + d\theta) = z_i(x_\delta; \theta) + \sum_{j=0}^{n_f} dW_{ij} \left[ \phi_j(x_\delta) + \epsilon \sum_{j_1=0}^{n_f} W_{ij_1} \psi_{j_1 j}(x_\delta) \right] + \frac{\epsilon}{2} \sum_{j_1, j_2=0}^{n_f} dW_{ij_1} dW_{ij_2} \psi_{j_1 j_2}(x_\delta).
$$

Let us make a shorthand for the quantity in the square bracket,

$$
\textcolor{blue}{\phi^E_{ij}(x_\delta; \theta)} = \frac{dz_i(x_\delta; \theta)}{dW_{ij}} = \phi_j(x_\delta) + \varepsilon \sum_{k=0}^{n_f} W_{ik} \psi_{kj}(x_\delta),
$$

which is a blue{effective feature function}.

### Effective Feature Functions

The utility of this is as follows:

- The *linear response* of $ z_i(x_\delta; \theta) $ behaves *effectively* as if it has a parameter-dependent feature function, $ \phi^E_{ij}(x_\delta; \theta) $.
- The change in the $ \phi^E_{ij}(x_\delta; \theta) $ given $ W_{ik} \to W_{ik} + dW_{ik} $ is

$$
\phi^E_{ij}(x_\delta; \theta + d\theta) = \phi^E

