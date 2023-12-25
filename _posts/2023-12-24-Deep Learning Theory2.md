---
title:  "Deep Learning Theory2"
mathjax: true
layout: post
categories: media
header-includes:
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{amsthm}
output: pdf_document

---


# deeplearningtheory

**Author:** Hyunin Lee  
**Date:** September 2023


## Day2
Lecture 2 is about Quadratic models and nearly-kernel methods. Chapter 11.4, chapter7.2 and chapter ∞.2.2 covers

### Linear Models and Kernel Methods

Two forms of a solution for a linear model:

- parameter space - linear regression

$$
z_i(x_{\dot{\beta}}; \theta^*) = \sum_{j=0}^{n_f} W_{ij}^* \phi_j(x_{\dot{\beta}})
$$

- sample space - kernel methods

$$
z_i(x_{\dot{\beta}}; \theta^*) = \sum_{\tilde{\alpha}_1, \tilde{\alpha}_2 \in A} k_{\dot{\beta} \tilde{\alpha}_1} \tilde{k}^{\tilde{\alpha}_1 \tilde{\alpha}_2} y_{i;\tilde{\alpha}_2}
$$


### Nonlinear models

Let's relax the above linear model into a nonlinear model, specifically a blue{quadratic model}.

$$
z_{i;\delta}(\theta) = \sum_{j=0}^{n_f} W_{ij} \phi_j(x_\delta) + \textcolor{blue}{\frac{\epsilon}{2} \sum_{j_1, j_2 = 0}^{n_f} W_{i j_1} W_{i j_2} \psi_{j_1 j_2}(x_\delta)}
$$

- It's nonlinear because it's quadratic in the weights: $W_{ij_1} W_{ij_2}$.
- $\varepsilon$ is a small parameter that controls the size of the deformation.
- We've introduced $\frac{(n_f + 1)(n_f + 2)}{2}$ meta feature functions, $\psi_{j_1 j_2} (x)$, with two feature indices.

### Quadratic models

To familiarize ourselves with this model, let's make a small change in the model parameters $W_{ij} \to W_{ij} + dW_{ij}$:

$$
z_i(x_\delta; \theta + d\theta) = z_i(x_\delta; \theta) + \sum_{j=0}^{n_f} dW_{ij} \left( \phi_j(x_\delta) + \epsilon \sum_{j_1=0}^{n_f} W_{ij_1} \psi_{j_1 j}(x_\delta) \right) + \frac{\epsilon}{2} \sum_{j_1, j_2=0}^{n_f} dW_{ij_1} dW_{ij_2} \psi_{j_1 j_2}(x_\delta)
$$

Let us make a shorthand for the quantity in the square bracket,

$$
\textcolor{blue}{\phi^E_{ij}(x_\delta; \theta)} = \frac{dz_i(x_\delta; \theta)}{dW_{ij}} = \phi_j(x_\delta) + \varepsilon \sum_{k=0}^{n_f} W_{ik} \psi_{kj}(x_\delta),
$$

which is a blue{effective feature function}.

### Effective Feature Functions

The utility of this is as follows:

- The *linear response* of $z_i(x_\delta; \theta)$ behaves *effectively* as if it has a parameter-dependent feature function, $\phi^E_{ij}(x_\delta; \theta)$.
- The change in the $\phi^E_{ij}(x_\delta; \theta)$ given $W_{ik} \to W_{ik} + dW_{ik}$ is

$$
\phi^E_{ij}(x_\delta; \theta + d\theta) = \phi^E
$$

### Quadratic Regression

Supervised learning a quadratic model doesn't have a particular name, but if it did, we'd all probably agree that its name should be quadratic regression:

$$
L_A(\theta) = \frac{1}{2} \sum_{\tilde{\alpha} \in A} \sum_{i=1}^{n_{out}} \left[ y_{i;\tilde{\alpha}} - \sum_{j=0}^{n_f} W_{ij} \phi_j(x_{\tilde{\alpha}}) - \frac{\epsilon}{2} \sum_{j_1, j_2 = 0}^{n_f} W_{ij_1} W_{ij_2} \psi_{j_1 j_2}(\tilde{x}_{\alpha}) \right]^2.
$$

The loss is now quartic in the parameters, but we can optimize with gradient descent:

$$
W_{ij}(t + 1) = W_{ij}(t) - \eta \frac{\partial L_A}{\partial W_{ij}} |_{W_{ij}=W_{ij}(t)}.
$$

This will find a minimum in practice.

#### The Theoretical Minimum
Let's start by seeing how gradient descent solves the *linear model*:

$$
L_A(W) = \frac{1}{2} \sum_{\tilde{\alpha} \in A} \sum_{i=1}^{n_{out}} \left[y_{i;\tilde{\alpha}} - \sum_{j=0}^{n_f} W_{ij} \phi_j(x_{\tilde{\alpha}}) \right]^2,
$$

Then, we have

$$
\begin{align*}
\frac{\partial L_A(W)}{\partial W_{ab}} &= - \sum_{\tilde{\alpha}, i, j} \delta_{ia}\delta_{jb} \phi_j(x_{\tilde{\alpha}}) \left[ y_{i;\tilde{\alpha}} - \sum_{j=0}^{n_f} W_{ij} \phi_j(x_{\tilde{\alpha}}) \right] \\
&= \sum_{\tilde{\alpha}} \phi_b(\tilde{x}_{\alpha}) (z_{a;\tilde{\alpha}} - y_{a;\tilde{\alpha}}) \\
&= \sum_{\tilde{\alpha}} \phi_b(\tilde{x}_{\alpha}) \epsilon_{a;\tilde{\alpha}}
\end{align*}
$$


In the last line, we defined the *residual training error*:

$$
\textcolor{blue}{\epsilon_{i;\tilde{\alpha}}} = z_{i;\tilde{\alpha}} - y_{i;\tilde{\alpha}}.
$$

The weights will update as
$$
\begin{align*}
    W_{ij}(t + 1) &= W_{ij}(t) - \eta \frac{d\mathcal{L}}{dW_{ij}} \Bigg|_{W_{ij}=W_{ij}(t)} \\ 
    &=W_{ij}(t) - \eta \sum_{\tilde{\alpha}} \phi_j(x_{\tilde{\alpha}}) \epsilon_{i;\tilde{\alpha}}(t)
\end{align*}
$$
For the theoretical analysis, it’s more convenient to understand how the output of the model updates:
$$
\begin{align*}
    z_{i;\delta}(t + 1) &= z_{i;\delta}(t) + \sum_{a,b} \frac{\partial z_{i;\delta}(t)}{\partial W_{ab}} \left[ W_{ab}(t + 1) - W_{ab}(t) \right] \\ 
    &= z_{i;\delta}(t) + \sum_{a,b} \frac{\partial z_{i;\delta}(t)}{\partial W_{ab}} \left[  - \eta \sum_{\tilde{\alpha}} \phi_b(x_{\tilde{\alpha}}) \epsilon_{a;\tilde{\alpha}}(t)  \right] \\ 
    &= z_{i;\delta}(t) + \sum_{a,b}
\end{align*}
$$

