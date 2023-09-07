eeee---
title:  "Deep learning theory 2.Neural Network"
mathjax: true
layout: post
categories: media
---

[Go to notion page](https://www.notion.so/Chapter2-Neural-Networks-b4af1c7431e84fa6b883729a58fdf6d4)


# Neural Networks 

In order to approximate an unknown non-linear function with neural network, it requires to learn (or estimate) the parameters $W^{(l)}_{i,j}$, $b^{(l)}_{i}$. This gives a hindsight on __how to initialize the parameters distribution__ is significant for the neural networks. 

## Initialization distribution of biases and weights

Usually, the obvious choice is the Gaussian distribution

$$\mathbb{E} [b^l_{i_{1}} b^l_{i_{1}}] = \delta_{i_1, i_2} C_{b}^{(l)}$$

$$\mathbb{E} \left[ W^{(l)}_{i_1,j_1} W^{(l)}_{i_2,j_2} \right] = \delta_{i_1 i_2} \delta_{j_1 j_2} \frac{C^{(l)}_W}{n_{l-1}}$$
