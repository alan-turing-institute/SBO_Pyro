"""
This is an implimentation of the structured bayesian optimisation approach based on
BOAT: Building Auto-Tuners with Structured Bayesian Optimization
Valentin Dalibard, Michael Schaarschmidt, Eiko Yoneki

"""

import copy
import numpy as np

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
            self.contstraints.append(
                constraints.interval(range_el[0], range_el[1]))


    def eval(self, x):
        """
        Evaluates target function any set of input points
            x: input points
        """

        raise NotImplementedError


normal_phi = lambda x: torch.exp(-x.pow(2) / 2) / np.sqrt(2 * np.pi)
normal_Phi = lambda x: (1 + torch.erf(x / np.sqrt(2))) / 2


class SemiParametricModel(PyroModule):

    def __init__(self, X, y, parametric_mean, kernel, jitter=1e-06):
        """ Defines a semi-parametric model, where the `parametric_mean` is a `PyroModule` """
        super().__init__()

        self.X = X
        self.y = y

        self.parametric_mean = parametric_mean

        self.kernel = kernel
        self.gp = gp.models.GPRegression(X, y, self.kernel, jitter=jitter)

    @pyro.nn.pyro_method
    def model(self):
        # Model definition code: explicitly subtract out the parametric mean, then call gp.model
        self.gp.set_data(self.X, self.y - self.parametric_mean(self.X))
        return self.gp.model()

    def forward(self, X):
        ''' Predict on new data points '''

        # reset "data" of GP to reflect mean estimate
        self.gp.set_data(self.X, self.y - self.parametric_mean(self.X))

        # sample mu, sigma
        mu, sigma = self.gp(X)

        # sample value of y
        y_hat = mu + self.parametric_mean(X)
        pyro.sample('y', dist.Normal(y_hat, sigma))

        # compute expected improvement
        y_min = self.y.min()
        delta = y_min - y_hat
        EI = delta.clamp_min(0.0) + sigma * normal_phi(
            delta / sigma) - delta.abs() * normal_Phi(delta / sigma)

        pyro.sample('EI', dist.Delta(-EI))

        # return the mean, in case we want to ignore the GP noise for some reason later
        return y_hat


def train(model, optimizer, loss, num_steps=1000):
    """ Trains the semi-parametric model. """

    # Autoguide
    guide = autoguide.AutoMultivariateNormal(model.model)

    svi = SVI(model.model, guide, optimizer, loss)

    # do gradient steps
    losses = []
    for _ in range(num_steps):
        losses.append(svi.step())

    return guide, losses


def transf_values(values, constr, dims, inv_mode=False):
    """ Transforming (un)constrained variables to (un)constrained domain """

    x_tmp = ()
    for i in range(dims):
        if inv_mode:
            x_tmp += (transform_to(constr[i]).inv(values[:, i]),)
        else:
            x_tmp += (transform_to(constr[i])(values[:, i]),)

    x = torch.stack(x_tmp, dim=1)

    return x


def find_a_candidate(model_predict, return_site, optimizer, x_init, constr, num_steps=1000, num_samples=5, lr=0.1):
    """ Finds new candidate """

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


def next_x(model_predict, return_site, optimizer, target, num_steps=1000, num_candidates=5, num_samples=5, lr=0.1):
    """ Finds the next best candidate on the acquisition function surface """

    candidates = []
    values = []

    # starts with the best candidate so far
    min_y = model_predict(model_predict.model.X)[return_site].mean(0)

    x_init = (model_predict.model.X[torch.min(min_y, dim=0)[1].item()]).unsqueeze(dim=0)

    for _ in range(num_candidates):
        x_can = find_a_candidate(model_predict, return_site,
                                 optimizer, x_init,
                                 target.contstraints,
                                 num_steps=num_steps,
                                 num_samples=num_samples,
                                 lr=lr)

        y = model_predict(x_can)[return_site].mean(0)

        candidates.append(x_can)
        values.append(y.detach())

        # a new random attempt initial point
        for _ in range(100):

            x_list = []
            for i in range(x_can.shape[-1]):
                x_list.append(x_can[:, i].new_empty(1).uniform_(
                    target.ranges[i][0], target.ranges[i][1]))

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


def update_posterior(model, optimizer, loss, obj_function, x_new, num_steps=1000):
    """
    Updates
        model:
        optimizer:
        loss:
        obj_function: objectivefunction
        x_new: a new value of x
        num_steps:

    """

    # evaluate f at new point
    bh_y = obj_function.eval(x_new)

    # incorporate new evaluation
    X = torch.cat([model.X, x_new])
    y = torch.cat([model.y, bh_y])

    model.X = X
    model.y = y

    # TODO: Check if this necessary
    model.gp.set_data(X, y)

    guide, losses = train(model, optimizer, loss, num_steps=num_steps)

    return guide, losses


def step(model,
         guide,
         optimizer,
         loss,
         target,
         acqf_optimizer,
         opti_num_steps=1000,
         acqf_opti_num_steps=1000,
         acqf_opti_lr=0.1,
         num_samples=1,
         num_candidates=5,
         return_site='EI'):
    """
    Performs a bayesian optimisation step. This includes generating a predictive model,
        which is used to find a new value of x at which the target function needs to be
        evaluated, and updating the model after the new x is obtained.

        model:
        guide:
        optimizer:
        loss:
        target:
        acqf_optimizer:
        opti_num_steps:
        acqf_opti_num_steps:
        num_samples:
        num_candidates:
        return_site:
    """

    if guide is not None:

        # TODO: copy.copy is a hack around predictive model being linked with a model
        # Constructs predictive distribution
        predict = pyro.infer.Predictive(copy.copy(model), guide=guide,
                                        num_samples=num_samples, return_sites=('y', return_site))

        # Finds the next candidate
        x_new = next_x(predict, return_site, acqf_optimizer, target,
                        num_steps=acqf_opti_num_steps, num_candidates=num_candidates,
                        num_samples=num_samples, lr=acqf_opti_lr)

        # Updates posterior
        new_guide, losses = update_posterior(model, optimizer, loss, target,
                                            x_new, num_steps=opti_num_steps)
    else:
        predict = None

        # trains the model
        new_guide, losses = train(model, optimizer, loss, num_steps=opti_num_steps)

    return new_guide, predict, losses
