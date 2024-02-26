---
title: Inverse Reinforcement Learning for Team Sports - Valuing Actions and Players
subtitle: A method that alternates single-agent IRL to learn a reward function for multiple agents.
layout: default
date: 2024-02-13
keywords: machine learing, reinforcement learning, player evaluation. inverse reinforcement learning
published: true
---

# Inverse Reinforcement Learning for Team Sports: Valuing Actions and Players
Yudong Luo, Oliver Schulte and Pascal Poupart

## rl vs irl

reinforcement learning vs inverse reinforcement learning

- RL: The primary goal of RL is for the agent to learn an optimal policy by interacting with the environment to maximize cumulative rewards.
- IRL: In contrast, the goal of IRL is to infer the underlying reward function from observed behavior or expert demonstrations, rather than learning a policy directly.

## abstract

sparse explicit reward 

hockey & soccer

combines **Q-function learning** with **inverse reinforcement learning (IRL)**

alternates single-agent IRL → reward function for multiple agents

knowledge transfer → combine learned rewards & observed rewards

## introduction

sparse explicit reward

1. Q-values show little variance
    
    → inverse reinforcement learning w/ domain knowledge (IRL-DK)
    
2. performance evaluation biased towards offensive players

**IRL-DK**

IRL → optimizing unobserved internal reward function

demonstrations & domain knowledge

**alternating learning** 

single-agent IRL for multi-agent Markov games

**procedure**:

- treat B as A’s environment
- learns a reward function for A in single-agent MDP
- flip the role and repeat the process

**contribution**

1. IRL for multi-agent dynamics
2. transfer learning for combining
    1. sparse explicit rewards
    2. learned dense implicit reward
3. alternating procedure

## Markov Game Model for Ice Hockey

### Markov Games and Decision Processes

games extend mdp

mdp is a single-agent markov game w/ `k=1`

policy {% katexmm %} $\pi_i = S \to PD(A_i)$ {% endkatexmm %}

action probability is a function of game state

actions are independent given current game state

game value function {% katexmm %} $G^{\pi_i, \pi_{i-1}}_i(s)$ {% endkatexmm %}

home & away as two agents → same action space

## Alternating Learning for Multi-Agent IRL

**marginal MDP $M(\pi_{-i}) \coloneqq \langle S, A_i, r^\prime, \gamma, T^\prime \rangle$** where

{% katexmm %}
$$
r^\prime(s,a_i) = \sum_{a_{-i}}r_i(s,a_i,a_{-i}) \cdot \pi_{-i}(a_{-i}|s) \\
T^\prime(s^\prime|a_i,s) = \sum_{a_{-i}}T(s^\prime|a_i, a_{-i}, s) \cdot \pi_{-i}(a_{-i}|s)
$$
{% endkatexmm %}

## IRL with Domain Knowledge

### maxent irl

maximum entropy IRL → interpretable linear model 

reward for trajectory $\zeta$ is cumulative reward of visited states

{% katexmm %}
$$
r(\zeta) = \sum_{s_j \in \zeta} \theta^T f_{s_j} = \theta^T f_\zeta
$$
{% endkatexmm %}

with reward weights $\theta \in \mathbb{R}^k, r_\theta(s) = \theta^Tf_s$

probability of demonstrated trajectory $\zeta$ increases exponentially with higher rewards 

if follows max entroy

probability can be estimated by 

{% katexmm %}
$$
P(\zeta | \theta, T) = \frac{e^{r_\zeta}}{Z(\theta, T)} \Pi_{s_{t+1}, a_t, s_t \in \zeta} P_T(s_{t+1}|a_t,s_t)
$$
{% endkatexmm %}

where {% katexmm %} $Z(\theta, T)$ {% endkatexmm %} is partition function and T is transition distribution

### maxent irl w/ dk

maxent irl fails to learn importance of goals due to sparsity

**rule reward function** assigns 1 for goal & 0 for others

want to

combine rule reward w/  maxent irl

{% katexmm %}
$$
\hat{\theta} = \argmax_\theta L(\theta) + \lambda k(r_\theta, r_K) \\
\text{where } r_\theta = \theta^T \psi, r_K = \theta^T_K \psi, \psi = [f_{s_1}, \dots, f_{s_n}] \in \mathbb{R}^{k \times n}
$$
{% endkatexmm %}

is the state feature matrix

mmd - maximum mean discrepancy

measurement of the difference between two probability distribution

unbiased estimation of squared MMD 

{% katexmm %}
$$
\begin{align*}
\hat{d}^2_{\mathcal{H}_k}(X,Y) &= \frac{1}{n^2_x} \sum^{n_x}_{i=1}\sum^{n_x}_{j=1}k(x_i,x_j) + \frac{1}{n^2_y}\sum^{n_y}_{i=1}\sum^{n_y}_{j=1}k(y_i,y_j) \\ 
&- \frac{2}{n_x n_y}\sum^{n_x}_{i=1}\sum^{n_y}_{j=1}k(x_i,y_j)
\end{align*}
$$
{% endkatexmm %}

where {% katexmm %} $k(x,x^\prime)$ {% endkatexmm %} is a kernel function 

the optimal {% katexmm %} $\hat{\theta}$ {% endkatexmm %} is derived by

{% katexmm %}
$$
\begin{align*}
\hat{\theta} &= \argmax_\theta L(\theta) - \alpha\hat{d}^2_{\mathcal{H}_k}(R_\theta, R_K) \\
&= \argmax_\theta L(\theta) + 2\alpha k(r_\theta, r_K)
\end{align*}
$$
{% endkatexmm %}

## evaluating learned reward and policy

### reward density

goal is to complement sparse reward

to cover many situations 

want

variance of learned rewards to be substantially higher than goal rewards

reflected in results

### policy evaluation

to evaluate how well the reward function rationalizes player behavior

best performing one yet

### learning performance

with regularization, way more stable and converges 

## player ranking

### action impact values

difference made by an action

{% katexmm %}
$$
\text{impact}_{\{H,A\}}(s,a) \equiv Q^{\pi^{\hat{\theta}}_{\{H,A\}}}_{\{H,A\}}(s,a)-V^{\pi^{\hat{\theta}}_{\{H,A\}}}_{\{H,A\}}(s)
$$
{% endkatexmm %}