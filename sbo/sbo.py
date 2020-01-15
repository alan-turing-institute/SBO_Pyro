"""
This is an implimentation of the structured bayesian optimisation approach based on 
BOAT: Building Auto-Tuners with Structured Bayesian Optimization
Valentin Dalibard, Michael Schaarschmidt, Eiko Yoneki

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

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
        ranges: range for each dimension
        
        """
        
        self.ranges = ranges
        
        self.contstraints = []
        
        for range_el in ranges:
            self.contstraints.append(
                constraints.interval(range_el[0], range_el[1]))
          
    def eval(self, x):
        return None

normal_phi = lambda x: torch.exp(-x.pow(2)/2)/np.sqrt(2*np.pi)
normal_Phi = lambda x: (1 + torch.erf(x / np.sqrt(2))) / 2

class SemiParametricModel(PyroModule):

    def __init__(self, X, y, parametric_mean, kernel):
        """ Defines a semi-parametric model, where the `parametric_mean` is a `PyroModule` """
        super().__init__()
                
        self.X = X
        self.y = y
        
        self.parametric_mean = parametric_mean
        
        self.kernel = kernel
        self.gp = gp.models.GPRegression(X, y, self.kernel)

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
        EI = delta.clamp_min(0.0) + sigma*normal_phi(delta/sigma) - delta.abs()*normal_Phi(delta/sigma)
        
        pyro.sample('EI', dist.Delta(-EI))
        
        # return the mean, in case we want to ignore the GP noise for some reason later
        return y_hat

def plot2D_obj(func, func_ranges, steps=100, strides=200):
    """Plots contour plot of a 2D function"""

    X1 = torch.linspace(func_ranges[0][0], func_ranges[0][1], steps)
    X2 = torch.linspace(func_ranges[1][0], func_ranges[1][1], steps)

    X1_mesh, X2_mesh = torch.meshgrid(X1, X2)
    
    Z_mesh = func(torch.stack((X1_mesh.flatten(), X2_mesh.flatten()), dim=1)).reshape(steps, steps)

    plt.contourf(
        X1_mesh.detach().numpy(), 
        X2_mesh.detach().numpy(), 
        Z_mesh.detach().numpy(), strides)
    
    plt.colorbar()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, optimizer, loss, num_steps=1000):
    """ Trains the semi-parametric model. """

    # Autoguide
    guide = autoguide.AutoMultivariateNormal(model.model)

    svi = SVI(model.model, guide, optimizer, loss)
    
    # do gradient steps
    losses = []
    for _ in range(num_steps):
        losses.append(svi.step())
    
    return losses, guide

def transf_values(values, constr, dims, inv_mode=False):
    """ Transforming (un)constrained variables to (un)constrained domain """
    
    x_tmp = ()
    for i in range(dims):
        if inv_mode:
            x_tmp += (transform_to(constr[i]).inv(values[:, i]), )
        else:
            x_tmp += (transform_to(constr[i])(values[:, i]), )
        
    x = torch.stack(x_tmp, dim=1)

    return x

def find_a_candidate(model_predict, return_site, x_init, constr, num_steps=1000, lr=0.1, num_samples=5):
    """ Finds new candidate """
      
    x_dims = x_init.shape[-1]
    
    x_uncon_init = transf_values(x_init, constr, x_dims, inv_mode=True)
    x_uncon = x_uncon_init.clone().detach().requires_grad_(True)
    
    # TODO: at the moment we are using torch optimizer, should we change to pyro?
    #     unconstrained minimiser 
    minimizer = optim.Adam([x_uncon], lr=lr)
    
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

def next_x(model_predict, return_site, target, num_candidates=5, num_steps=1000, lr=0.1, num_samples=5):
    """ Finds the next best candidate on the acquisition function surface """
    
    candidates = []
    values = []
    
    # start with the last step
    x_init = model_predict.model.X[-1:]

    for _ in range(num_candidates):

        x_can = find_a_candidate(model_predict, return_site, x_init, target.contstraints, 
                                 num_steps=num_steps, lr=lr, num_samples=num_samples)
        
        y = model_predict(x_can)[return_site].mean(0)
        
        candidates.append(x_can)
        values.append(y)
        
        # a new random attempt initial point
        for _ in range(100):

            values = []
            for i in range(x_can.shape[-1]):
                values.append(x_can[:,i].new_empty(1).uniform_(target.ranges[i][0], target.ranges[i][1]))
            x_init = torch.stack(values, dim=1)

            y_init = model_predict(x_init)[return_site].mean(0)
            if y_init < 0.0:
                break  
        
    argmin = torch.min(torch.cat(values), dim=0)[1].item()
        
    return candidates[argmin]

def update_posterior(model, obj_function, x_new, num_steps=1000, adam_params={"lr":0.1}):
    
    # evaluate f at new point
    bh_y = obj_function(x_new) 
        
    # incorporate new evaluation
    X = torch.cat([model.X, x_new]) 
    y = torch.cat([model.y, bh_y])
    
    model.X = X
    model.y = y
    
    # TODO: Check if this necessary
    model.gp.set_data(X, y)
    
    losses, guide = train(model, num_steps=num_steps, adam_params=adam_params)
    
    return guide, losses
