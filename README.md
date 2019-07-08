# DPM
Differentiable Probabilistic Models

---

## TODO:
  * Implement ChiSquare Distribution
  * Finish VAE example in Notebook
  * GAN Example using Distribution Components
  * GLM Models from TensorFlow
  * Add more Transforms from TensorFlow
  * Fix Uniform low/high switching
  * Fix Softsign test issue of NaN

---

## Table of Contents
1. [Distributions](#distributions)
2. [Transforms](#transforms)
3. [Mixture Models](#mixture-models)
4. [Divergences](#divergences)
5. [Adversarial Loss](#adversarial-loss)
6. [ELBO](#elbo)
7. [MCMC Methods](#mcmc-methods)
8. [Notes](#notes)

---

# 1. Distributions <a name="distributions"></a>
  1. [Normal (Multivariate)](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
  2. [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
  3. [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
  4. [Beta](https://en.wikipedia.org/wiki/Beta_distribution)
  5. [Log Normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
  6. [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
  7. [Relaxed Bernoulli](https://arxiv.org/abs/1611.00712)
  8. [Gumbel Softmax (Relaxed Categorical)](https://arxiv.org/abs/1611.01144)
  9. [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
  10. [Student T](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
  11. [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
  12. [Fisher-Snedecor (F-Distribution)](https://en.wikipedia.org/wiki/F-distribution)
  13. [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function)
  14. Data Distribution
      * Randomly sample from a given set of data.
  15. Half Cauchy
  16. Half Normal
  17. [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
  18. Conditional Model
      * Uses a Neural Network to take inputs and create the parameters of a distribution.
      * Sampling -> takes a value, runs the network to create the distribution,
        sample from conditional distribution.
      * Log Prob -> Create distribution conditioned on X, take log_prob w.r.t. Z
  19. Transform Distribution
      * Composes a list of [transforms](#transforms) on a distribution
      * Example: Exp(Normal) ~ LogNormal
  21. [Convolution](https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions)
      * Sum of component distributions, only allows sampling
  20. ChiSquare (TODO)

# 2. Transforms <a name="transforms"></a>
  1. Exp
  2. Power
  3. Reciprocal
  4. Sigmoid
  5. Affine
  6. Expm1
  7. Gumbel
  8. SinhArcsinh
  9. Softplus
  10. Softsign
  11. Tanh

# 3. Mixture Model <a name="mixture-model"></a>
  1. [Mixture Model](https://en.wikipedia.org/wiki/Mixture_model)
      * Static weights to pick from sub-models using a categorical distribution.
  2. [Gumbel Mixture Model](https://arxiv.org/abs/1611.01144)
      * Uses the Gumbel Softmax as a differentiable approximation to the
      categorical distribution, allowing mixture weights to be learned.
  3. Infinite Mixture Model
      * Student T written as a Mixture Model.

# 4. Divergences <a name="divergences"></a>
  1. Cross Entropy
  2. [Forward KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
      * P Model -> Sampling (rsample)
      * Q Model -> PDF Function (log_prob)
  3. [Reverse KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
      * P Model -> PDF Function (log_prob)
      * Q Model -> Sampling + PDF Function
  4. [Jensen-Shannon Divergence (JS)](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
      * P Model -> PDF + Sampling
      * Q Model -> PDF + Sampling

# 5. Adversarial Loss <a name="adversarial-loss"></a>
  1. [Adversarial Loss (aka GAN Loss)](https://arxiv.org/pdf/1711.10337.pdf)
      * Hides a discriminator under the loss function, and computes the adversarial loss
      * P Model -> Sampling (rsample)
      * Q Model -> Sampling (rsample)
      * Must use RMSprop
      * Algorithms:
          1. [GAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
          2. [MMGAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
          3. [WGAN Loss](https://arxiv.org/pdf/1701.07875.pdf)
          4. [LSGAN Loss](https://arxiv.org/pdf/1611.04076.pdf)

# 6. ELBO <a name="elbo"></a>
  1. Implements SVI with ELBO loss.
  2. Requires a Conditional Model to learn, in addition to P and Q models.

# 7. MCMC Methods <a name="mcmc-methods"></a>
  1. [Metropolis–Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)
      * MCMC sampling method to generate samples from a unknown distribution
        * Requires distribution to have a log_prob method implemented.

# 8. Notes <a name="notes"></a>
  * Sampling must be done through a reparameterized version of the
    distribution to allow gradients to back-prop through samples.
  * Probabilities are in log form, for numerical stability.
