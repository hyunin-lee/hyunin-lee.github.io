---
title:  "Advanced examples"
mathjax: true
layout: post
categories: media
---

![Swiss Alps](https://user-images.githubusercontent.com/4943215/55412536-edbba180-5567-11e9-9c70-6d33bca3f8ed.jpg)


## MathJax

Dear Reviewer 4AXz,

Thanks for the fast response! 
We appreciate the reviewer's effort to correct the misunderstandings. Thanks to the reviewer's detailed comments, we are able to provide more specific explanations and corrections based on your last comments. __We would love to hear back whether other concerns ( 2),3),4) of your last comment) are solved, or it not, we like to provide further clarification.__

First, we divide your first paragraph (comment 1) into two subquestions as follows.

$\textbf{[1-1] Different timelines between real-time and framework. How do you apply to equation 2.1?}$

__We admit the time difference__ between the time elapsed in real-time (let's call it $t_{real}$) and the time elapsed in our framework (let's call it $t_{frame}$). Let's also assume the sampling time for one episode is $\Delta_{sp}$ and training time is $\Delta_{t}$. Let's start the system at  $t_{real}=t_{frame}=0$ which agent immediately starts 1st episode. As you mentioned, then the time that agent starts 2nd episode would be $t_{real}= \Delta_{sp} + \Delta_{t}$ but $t_{frame}= \Delta_{t}$, and 3rd episode would be $t_{real}= 2\Delta_{sp} + 2\Delta_{t}$ but $t_{frame}= 2\Delta_{t}$, then $k$th episode would be $t_{real}= (k-1)\Delta_{sp} + (k-1)\Delta_{t}$ but $t_{frame}= (k-1)\Delta_{t}$

__However, we can tune the time difference and applicable to Eq 2.1__ by setting $\Delta_{t} \leftarrow \Delta_{sp} + \Delta_{t}$. Namely, the agent recognize the time $t_{frame}$, not $t_{real}$, but if agent self-compensate $\Delta_{t} \leftarrow \Delta_{sp} + \Delta_{t}$, then it makes $t_{frame}=t_{real}$. __The agent is thoroughly able to do tuning by itself before starting 1st episode because $\Delta_{sp}$ is only proportional to step $H$ and $H$ is given as prior information. This means the agent can access (or calculate) $\Delta_{sp}$ as prior constant, and able to set $\Delta_{t} \leftarrow \Delta_{sp} + \Delta_{t}$ before starting a 1st episode.

$\textbf{[1-2]My doubt is: why can we consider the environment stationary during an episode and only non-stationary in between episodes?}$

If the above [1-1] was addressed, then let's focus on whether the existence of $\Delta_{sp} >0$ harms our framework and theoretical analysis. We admit that the real-time system can change during $\Delta_{sp}$, and situation that $t=[0,  \Delta_{sp}]$ is stationary, $t=[\Delta_{sp} , \Delta_{sp}+\Delta_{t}]$ is non-stationary does not make sense. 

__Foremost, if we add the assumption that__ $\Delta_{sp} / \Delta_{t} \approx 0 $, __then__ $\Delta_{sp}$ __can be ignored and perfectly matches with all of our work.__

__However, the point is__  $\Delta_{sp} / \Delta_{t} \not\approx 0 $ __does not harm our framework__. This is because if $\Delta_{sp} = 0$, then the agent learns an optimal policy from the observed environments $( \mathcal{M}[t_{real}=0],\mathcal{M}[t_{real}=\Delta_{t}],...)$, and if $\Delta_{sp} \neq 0$, then the agent learns from $( \mathcal{M}[t_{real}=0 \sim\Delta_{sp}],\mathcal{M}[t_{real}=\Delta_{t}+\Delta_{sp} \sim \Delta_{t} +2\Delta_{sp}],...)$. In both cases, essentially, the agent observed trajectories from non-stationary distributions that changes for every episode, and learn near-optimal policy from that observed trajectories.  __The only difference between two cases is data-sampling distributions which have different non-stationarity__ 

Now, if the environment is non-stationary during an episode, then it's a simple extension of our theoretical analysis. The difference occurs in how can we guarantee near-optimal policy since a trajectory is sampled from non-stationary distribution, not a stationary distribution. To address this issue, all we need to change is including step-varying environment change to the local variation budget (ex-$\sum_{k=1}^{K}\sum_{h=1}^{H} || P^{t_{(k,h+1)}}-P^{t_{(k,h)}}||_\infty$ ) , which is widely used definition for intra-changing episodes.[14,16,17,37]. __This does not fundamentally change the mathematical expression on all of the theoretical analysis but potentially provides a loose upper bound due to the looser terms __ $B_r(G),B_p(G)$.

$\textbf{[2] Misunderstanding on the total time elapsed}$

Thanks for re-raising this issue. We elaborate the robot example and correct misunderstanding. For fixed elapsed real-time duration $t_{real} \in [0,T]$, let's set the sampling time is $\Delta_{sp}$, and assume robot A's training policy time $\Delta_t (=\Delta_A)$ and that of robot B as $2\Delta_t (=\Delta_B)$. Then, robot A 's episode $K_A= \lfloor T / (\Delta_t + \Delta_{sp} ) \rfloor$ and robot B 's episode $K_B= \lfloor T / (\Delta_t + 2\Delta_{sp} ) \rfloor$. Now, the how many times to interact (=episdes) is different as $K_A > K_B$, but policy policy training time in inverse, i.e. $\Delta_A < \Delta_B$. We would like to emphasize that the __inverse relationship__ is the key source to raise the trade-off problem and shed a light on optimal training time, which leads to Proposition 3. 





You can enable MathJax by setting `mathjax: true` on a page or globally in the `_config.yml`. Some examples:

[Euler's formula](https://en.wikipedia.org/wiki/Euler%27s_formula) relates the  complex exponential function to the trigonometric functions.

$$ e^{i\theta}=\cos(\theta)+i\sin(\theta) $$

The [Euler-Lagrange](https://en.wikipedia.org/wiki/Lagrangian_mechanics) differential equation is the fundamental equation of calculus of variations.

$$ \frac{\mathrm{d}}{\mathrm{d}t} \left ( \frac{\partial L}{\partial \dot{q}} \right ) = \frac{\partial L}{\partial q} $$

The [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation) describes how the quantum state of a quantum system changes with time.

$$ i\hbar\frac{\partial}{\partial t} \Psi(\mathbf{r},t) = \left [ \frac{-\hbar^2}{2\mu}\nabla^2 + V(\mathbf{r},t)\right ] \Psi(\mathbf{r},t) $$

## Code

Embed code by putting `{{ "{% highlight language " }}%}` `{{ "{% endhighlight " }}%}` blocks around it. Adding the parameter `linenos` will show source lines besides the code.

{% highlight c %}

static void asyncEnabled(Dict* args, void* vAdmin, String* txid, struct Allocator* requestAlloc)
{
    struct Admin* admin = Identity_check((struct Admin*) vAdmin);
    int64_t enabled = admin->asyncEnabled;
    Dict d = Dict_CONST(String_CONST("asyncEnabled"), Int_OBJ(enabled), NULL);
    Admin_sendMessage(&d, txid, admin);
}

{% endhighlight %}

## Gists

With the `jekyll-gist` plugin, which is preinstalled on Github Pages, you can embed gists simply by using the `gist` command:

<script src="https://gist.github.com/5555251.js?file=gist.md"></script>

## Images

Upload an image to the *assets* folder and embed it with `![title](/assets/name.jpg))`. Keep in mind that the path needs to be adjusted if Jekyll is run inside a subfolder.

A wrapper `div` with the class `large` can be used to increase the width of an image or iframe.

![Flower](https://user-images.githubusercontent.com/4943215/55412447-bcdb6c80-5567-11e9-8d12-b1e35fd5e50c.jpg)

[Flower](https://unsplash.com/photos/iGrsa9rL11o) by Tj Holowaychuk

## Embedded content

You can also embed a lot of stuff, for example from YouTube, using the `embed.html` include.

{% include embed.html url="https://www.youtube.com/embed/_C0A5zX-iqM" %}
