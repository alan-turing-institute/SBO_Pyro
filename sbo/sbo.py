"""
This is an implimentation of the structured bayesian optimisation approach based on
BOAT: Building Auto-Tuners with Structured Bayesian Optimization by Dalibard et al.
and https://pyro.ai/examples/bo.html.

"""

import copy
import numpy as np
import math

import torch
import torch.nn as nn
from torch.distributions import constraints, transform_to
import torch.optim as optim
import torch.autograd as autograd

import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.nn import PyroSample, PyroModule
from pyro.infer import autoguide, SVI, Trace_ELBO


NORMAL_PHI_1 = lambda x: torch.exp(-x.pow(2) / 2) / np.sqrt(2 * np.pi)
NORMAL_PHI_2 = lambda x: (1 + torch.erf(x / np.sqrt(2))) / 2


class TargetFunction:
    """A class to represent target function"""

    def __init__(self, ranges):
        """
        Constructor for a target function - processess ranges into constrains.
            ranges: range for each dimension

        """

        self.ranges = ranges

        self.contstraints = []

        for range_el in ranges:
            self.contstraints.append(constraints.interval(range_el[0], range_el[1]))

    def eval(self, x):
        """
        Evaluates target function any set of input points
            x: input points
        """

        raise NotImplementedError


class SemiParametricModel(PyroModule):
    """
    Semi-parametric model class which extends PyroModule and employs Gaussian process
        regression model (GPRegression).
    """

    def __init__(self, X, y, parametric_mean, kernel, noise=None, jitter=1e-06):
        """ 
        Defines a semi-parametric model.

        Args:
            X (torch.Tensor) – A input data for training. Its first dimension is the number of data points.
            y (torch.Tensor) – An output data for training. Its last dimension is the number of data points.
            parametric_mean (callable) - A parametric mean function implemneted as PyroModule
            kernel (Kernel) – A Pyro kernel object, which is the covariance function 𝑘.
            noise (torch.Tensor) – Variance of Gaussian noise of this model.
            jitter (float) – A small positive term which is added into the diagonal 
                part of a covariance matrix to help stablize its Cholesky decomposition.
        """

        super().__init__()

        self.X = X
        self.y = y

        self.parametric_mean = parametric_mean

        self.kernel = kernel
        self.gp = gp.models.GPRegression(X, y, self.kernel, noise=noise, jitter=jitter)

    @pyro.nn.pyro_method
    def model(self):
        """
        A model stochastic function, where we explicitly subtract out the parametric mean
            from the training output data. This means that we model based on the difference
            (errors) between ParametricMeanFn and the observations (TargetFunction).

        Args:
            X (torch.Tensor) – A input data for training. Its first dimension is the number of data points.
            y (torch.Tensor) – An output data for training. Its last dimension is the number of data points.
            parametric_mean (callable) - A parametric mean function implemneted as PyroModule
            kernel (Kernel) – A Pyro kernel object, which is the covariance function 𝑘.
            noise (torch.Tensor) – Variance of Gaussian noise of this model.
            jitter (float) – A small positive term which is added into the diagonal 
                part of a covariance matrix to help stablize its Cholesky decomposition.
        Returns:
            A “model” stochastic function for a GP. 
        """

        # Model definition code: explicitly subtract out the parametric mean, then call gp.model
        self.gp.set_data(self.X, self.y - self.parametric_mean(self.X))
        
        return self.gp.model()

    def forward(self, X):
        """
        Computes the mean and covariance matrix (or variance) of Gaussian Process posterior on the
            difference data. Then the obtained mean value together with the parametric mean
            function are used to sample values for $y$, which in turn are used to sample the
            expected improvement values.

        Args:
            X (torch.Tensor) – A input data for training. Its first dimension is the number of data points.
        Returns:
            y_hat (float) - The mean, in case we want to ignore the GP noise for some reason later
        """

        # reset "data" of GP to reflect mean estimate
        self.gp.set_data(self.X, self.y - self.parametric_mean(self.X))

        # sample mu, sigma
        mu, sigma = self.gp(X)

        # sample value of y
        y_hat = mu + self.parametric_mean(X)
        pyro.sample("y", dist.Normal(y_hat, sigma))

        # compute expected improvement
        y_min = self.y.min()
        delta = y_min - y_hat
        
        EI = (
            delta.clamp_min(0.0)
            + sigma * NORMAL_PHI_1(delta / sigma)
            - delta.abs() * NORMAL_PHI_2(delta / sigma)
        )

        # sample the expected improvement
        pyro.sample("EI", dist.Delta(-EI))

        return y_hat


class GPRegressionModule(PyroModule):
    """
    A model class which extends PyroModule and employs Gaussian process
        regression model (GPRegression).
    """

    def __init__(self, X, y, kernel):
        """ Defines a PyroModule which wraps GPRegression 

         Args:
            X (torch.Tensor) – A input data for training. Its first dimension is the number of data points.
            y (torch.Tensor) – An output data for training. Its last dimension is the number of data points.
            kernel (Kernel) – A Pyro kernel object, which is the covariance function 𝑘.
        """

        super().__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.gp = gp.models.GPRegression(X, y, self.kernel)

    @pyro.nn.pyro_method
    def model(self):
        """
        A model stochastic function.

        Returns:
            A “model” stochastic function for a GP. 
        """

        return self.gp.model()

    def forward(self, X):
        """
        Computes the mean and covariance matrix (or variance) of Gaussian Process posterior. Then the 
            obtained mean value is used to sample values for $y$, which in turn are used to sample the
            expected improvement values.

        Args:
            X (torch.Tensor) – A input data for training. Its first dimension is the number of data points.
        Returns:
            mu (float) - The mean, in case we want to ignore the GP noise for some reason later
        """

        # sample mu, sigma
        mu, sigma = self.gp(X)

        # sample value of y
        pyro.sample("y", dist.Normal(mu, sigma))

        # compute expected improvement
        y_min = self.y.min()
        delta = y_min - mu
        EI = (
            delta.clamp_min(0.0)
            + sigma * NORMAL_PHI_1(delta / sigma)
            - delta.abs() * NORMAL_PHI_2(delta / sigma)
        )

        pyro.sample("EI", dist.Delta(-EI))

        return mu


def train(model, optimizer, loss, num_steps=1000):
    """ 
    Trains a model. 
    
    Args:
        model (PyroModule instance) – the model
        optimizer (pyro.optim.PyroOptim) – a wrapper for a PyTorch optimizer.
        loss (pyro.infer.elbo.ELBO) – an instance of a subclass of ELBO.
        num_steps (int (1000)) - the number of optimization steps.
    Returns:
        guide (pyro.infer.autoguide.guides.AutoContinuous) - a MultivariateNormal posterior distribution.
        losses (list(floats)) - a list of losses during the training procedure
    """

    # Autoguide
    guide = autoguide.AutoMultivariateNormal(model.model)

    svi = SVI(model.model, guide, optimizer, loss)

    # do gradient steps
    losses = []
    for _ in range(num_steps):
        losses.append(svi.step())

    return guide, losses


def transf_values(values, constr, dims, inv_mode=False):
    """ 
    Transforms constrained values to unconstrained domain and vice versa.
    
    Args:
        values (torch.Tensor) - values to be transformed.
        constr (list) - a list of constrains.
        dims (int) - a number of dimensions.
        inv_mode (bool) - a flag for the direction of transformation:
            False: constrained->unconstrained, True: constrained->unconstrained.
    Returns:
        x (torch.Tensor) - transformed values.
    """

    x_tmp = ()
    for i in range(dims):
        if inv_mode:
            x_tmp += (transform_to(constr[i]).inv(values[:, i]),)
        else:
            x_tmp += (transform_to(constr[i])(values[:, i]),)

    x = torch.stack(x_tmp, dim=1)

    return x


def find_a_candidate(model_predict, return_site, optimizer, x_init, constr,
                     num_steps=1000, lr=0.1):
    
    """ 
    Finds a new candidate point.
    
    Args:
        model_predict (pyro.infer.Predictive) - predictive distribution.
        return_site (list, tuple, or set) – sites to return.
        optimizer (torch.optim) - acquisition function optimizer.
        x_init (torch.Tensor) - initial optiisation point.
        constr (list) - list of constrains.
        num_steps (int (1000)) - number of acquisition function optimization steps.
        lr (float (0.1)) - learning rate for acquisition function optimization.
    Returns:
        x (torch.Tensor) - a new candidate point.
    """

    x_dims = x_init.shape[-1]

    x_uncon_init = transf_values(x_init, constr, x_dims, inv_mode=True)
    x_uncon = x_uncon_init.clone().detach().requires_grad_(True)

    minimizer = optimizer([x_uncon], lr=lr)

    def closure():
        minimizer.zero_grad()
        x = transf_values(x_uncon, constr, x_dims)

        y = model_predict(x)[return_site].mean(0)

        autograd.backward(x_uncon, autograd.grad(y, x_uncon))
        return y

    for _ in range(num_steps):
        minimizer.step(closure)

    x = transf_values(x_uncon, constr, x_dims)

    return x.detach()


def next_x(model_predict, return_site, optimizer, target,
           num_steps=1000, num_candidates=5, lr=0.1):
    """ 
    Finds the next best candidate on the acquisition function surface.
    
    Args:
        model_predict (pyro.infer.Predictive) - predictive distribution.
        return_site (list, tuple, or set) – sites to return.
        optimizer (torch.optim) - acquisition function optimizer.
        target (TargetFunction) - the target (black-box) function.
        num_steps (int (1000)) - number of acquisition function optimization steps.
        num_candidates (int (5)) - number of candidates to try during each optimisation step.
        lr (float (0.1)) - learning rate for acquisition function optimization.
    Returns:
        (torch.Tensor) - a new candidate point.
    """

    candidates = []
    values = []

    # starts with the best candidate so far
    min_y = model_predict(model_predict.model.X)[return_site].mean(0)

    x_init = (model_predict.model.X[torch.min(min_y, dim=0)[1].item()]).unsqueeze(dim=0)

    for _ in range(num_candidates):
        x_can = find_a_candidate(
            model_predict,
            return_site,
            optimizer,
            x_init,
            target.contstraints,
            num_steps=num_steps,
            lr=lr
        )

        y = model_predict(x_can)[return_site].mean(0)

        if not math.isnan(float(y.detach())):
            candidates.append(x_can)
            values.append(y.detach())

        # a new random attempt initial point
        for _ in range(100):

            x_list = []
            for i in range(x_can.shape[-1]):
                x_list.append(
                    x_can[:, i]
                    .new_empty(1)
                    .uniform_(target.ranges[i][0], target.ranges[i][1])
                )

            x_init = torch.stack(x_list, dim=1)

            y_init = model_predict(x_init)[return_site].mean(0)

            # since we are minimising, we check
            if y_init < 0.0:
                break

    if num_candidates > 1:
        argmin = torch.min(torch.cat(values), dim=0)[1].item()

        return candidates[argmin]
    else:
        return candidates[0]


def update_posterior(model, optimizer, loss, target, x_new, num_steps=1000):
    """
    Updates the postrior distribution.

    Args:
        model (PyroModule instance) – the model.
        optimizer (pyro.optim.PyroOptim) – a wrapper for a PyTorch optimizer.
        loss (pyro.infer.elbo.ELBO) – an instance of a subclass of ELBO.
        target (TargetFunction) - the target (black-box) function.
        x_new  (torch.Tensor) - a candidate point.
        num_steps (int (1000)) - the number of optimization steps.

    Returns:
        guide (pyro.infer.autoguide.guides.AutoContinuous) - an updated MultivariateNormal posterior distribution.
        losses (list(floats)) - a list of losses during the training procedure.
    """

    # evaluate f at new point
    obj_y = target.eval(x_new)

    # incorporate new evaluation
    X = torch.cat([model.X, x_new])
    y = torch.cat([model.y, obj_y])

    model.X = X
    model.y = y

    model.gp.set_data(X, y)

    guide, losses = train(model, optimizer, loss, num_steps=num_steps)

    return guide, losses


def step(model, guide, optimizer, loss, target, acqf_optimizer,
         opti_num_steps=1000, acqf_opti_num_steps=1000,
         acqf_opti_lr=0.1, num_samples=1, num_candidates=5,
         return_site="EI"):
    """
    Performs a bayesian optimisation step. This involves generating a predictive model, 
    which is then used to find a new value of x at which the target function needs to 
    be evaluated, and updating the model after the new x is obtained.

    Args:
        model (PyroModule instance) – the model
        guide (pyro.infer.autoguide.guides.AutoContinuous) - a MultivariateNormal posterior distribution.
        optimizer (pyro.optim.PyroOptim) – a wrapper for a PyTorch optimizer.
        loss (pyro.infer.elbo.ELBO) – an instance of a subclass of ELBO.
        target (TargetFunction) - the target (black-box) function
        acqf_optimizer (torch.optim) - the acquisition function optimizer.
        opti_num_step (int (1000)) - the number of inference optimization steps.
        acqf_opti_num_steps (int (1000)) - the number of acquisition function optimization steps.
        acqf_opti_lr (float (0.1)) - the learning rate for acquisition function optimization.
        num_samples (int (1)) - number of samples to draw from the predictive distribution.
        num_candidates (int (5)) - number of candidates to try during each optimisation step.
        return_site (list, tuple, or set ("EI")) – sites to return.

    Returns:
        new_guide (pyro.infer.autoguide.guides.AutoContinuous) - an updated MultivariateNormal posterior distribution.
        predict (pyro.infer.Predictive) - a predictive distribution.
        losses (list(floats)) - a list of losses during the training procedure.
    """

    if guide is not None:

        # TODO: copy.copy is a hack around predictive model being linked with a model
        # Constructs predictive distribution
        predict = pyro.infer.Predictive(
            copy.copy(model),
            guide=guide,
            num_samples=num_samples,
            return_sites=("y", return_site)
        )

        # Finds the next candidate
        x_new = next_x(
            predict,
            return_site,
            acqf_optimizer,
            target,
            num_steps=acqf_opti_num_steps,
            num_candidates=num_candidates,
            lr=acqf_opti_lr
        )

        # Updates posterior
        new_guide, losses = update_posterior(
            model, optimizer, loss, target, x_new, num_steps=opti_num_steps
        )
    else:
        predict = None

        # trains the model
        new_guide, losses = train(model, optimizer, loss, num_steps=opti_num_steps)

    return new_guide, predict, losses
