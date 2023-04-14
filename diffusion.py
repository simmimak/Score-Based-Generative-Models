import torch
import numpy as np
from scipy import integrate

class VariancePreservingSDE(torch.nn.Module):

    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)
    
    def marginal_prob_std(self, t):
       return self.var(t) ** 0.5 
       
    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)


error_tolerance = 1e-5
def ode_sampler(score_model,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3):
    
    T1 = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)
        
    sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T1)

    t = torch.ones(batch_size, device=device)

    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
        * sde.marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z
        
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))   

        with torch.no_grad():    
            score = score_model(sample, time_steps)
            return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, x):        
        time_steps = np.ones((shape[0],)) * t   

        g = sde.g(torch.tensor(t),np.ones((shape[0],))).cpu().numpy()

        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)


    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


def prior_likelihood(z, sigma):
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x, 
                   score_model,
                   batch_size=64, 
                   eps=1e-5):
    
    epsilon = torch.randn_like(x)
    device='cuda'
    T1 = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)
        
    sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T1)
  
    def divergence_eval(sample, time_steps, epsilon):      
      
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    
    
    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def divergence_eval_wrapper(sample, time_steps):
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)
        
    def ode_func(t, x):
        time_steps = np.ones((shape[0],)) * t    
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = sde.g(torch.tensor(t), sample).cpu().numpy()
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = sde.marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return z, bpd
