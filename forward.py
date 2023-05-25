import torch
import numpy as np
from torch.nn.functional import interpolate as interp

def get_sampler(args, device):
    if args.forward_name == 'toy':
        sampler = ToySampler(args, device)
    elif args.forward_name == 'ddpm':
        sampler = NoiseSampler(args, device)
    elif args.forward_name == 'vanilla_sr':
        sampler = VanillaSRSampler(args, device)
    elif args.forward_name == 'sr':
        sampler = SRSampler(args, device)
    else:
        raise NotImplementedError
        
    return sampler


class BaseSampler:
    def __init__(self, args, device):
        """
        Args:
            args (Class): The class should contain the followings: 
                num_timesteps (int): The number of total timesteps.
                nz (int): The dimension of latent vector z.
                num_channels (int): The number of image channels.
                image_size (int): Image size. (Only square images are allowed.)
        """
        self.args = args
        self.device = device
        self.T = args.num_timesteps
        self.num_channels = args.num_channels
        self.image_size = args.image_size
        self.nz = args.nz
        self._coeff()
        self.posterior = args.posterior
        if args.posterior:
            self._pos_coeff()
    
    def extract(self, input, t, shape):
        out = torch.gather(input, 0, t)
        reshape = [shape[0]] + [1] * (len(shape) - 1)
        out = out.reshape(*reshape)
        return out

    def _coeff(self):
        '''
        get coefficients related with forward process
        '''
        raise NotImplementedError
    
    def _pos_coeff(self):
        '''
        get posterior coefficients
        '''
        pass
    
    def sample(self, x0, t, noise=None, noise_factor=1.):
        '''
        sample t-level degraded data from original data x0
        Args:
            x0 (tensor) : input image tensor of shape (num, num_channels, image_size, image_size)
            t (tensor) : degradation level of shape (num,)
            noise (tensor) : input noise. shape should be same as x0
        '''
        raise NotImplementedError
    
    def pos_sample(self, x0, x_tp1, t, noise=None, noise_factor=1.):
        pass
    
    def sample_pairs(self, x0, t):
        '''
        sample t-level degraded image and (t+1)-level degraded image
        '''
        x_t = self.sample(x0, t)
        x_t_plus_one = self.sample(x0, t+1)
        return x_t, x_t_plus_one
    
    def fidelity(self, x0, t, x_t_plus_one):
        '''
        calculate fidelity loss.
        Args:
            x0 (tensor) : restored image tensor of shape (num, num_channels, image_size, image_size)
            t (tensor) : degradation level of shape (num,)
            x_t_plus_one (tensor) : (t+1)-level degraded image
        '''
        pass
    
    def init_sample(self, num):
        '''
        sample "num" numbers of initial distribution (Gaussian)
        '''
        raise NotImplementedError
    
    def sample_from_model(self, generator, x_init=None, num=None):
        '''
        generate "num" numbers of image from given "x_init" (the prior).
        generator is a trained generator
        '''
        if x_init is None:
            x_init = self.init_sample(num).to(self.device)
        with torch.no_grad():
            x = x_init
            for i in reversed(range(self.T)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                latent_z = torch.randn(x.size(0), self.nz, device=x.device)
                x_0 = generator(x, t, latent_z)
                if not self.posterior:
                    x = self.sample(x_0, t, noise_factor=0.999)
                else:
                    x = self.pos_sample(x_0, x, t)
        return x


class ToySampler(BaseSampler):
    def __init__(self, args, device):
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.discretize = args.discrete_type
        super().__init__(args, device)
             
    def _coeff(self):
        t = np.arange(0, self.T + 1, dtype=np.float64)
        t = t / self.T
        eps_small = 1e-3
        t = torch.from_numpy(t) * (1. - eps_small) + eps_small
        
        if self.discretize == 'geometric':
            var = var_func_geometric(t, self.beta_min, self.beta_max)
        elif self.discretize == 'square':
            var = var_func_vp(t**2, self.beta_min, self.beta_max)
        else:
            var = var_func_vp(t, self.beta_min, self.beta_max)
        
        alpha_bars = 1.0 - var
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        first = torch.tensor(1e-8)
        betas = torch.cat((first[None], betas)).to(self.device)
        self.betas = betas = betas.type(torch.float32)
        self.sigmas = betas**0.5
        self.a_s = torch.sqrt(1-betas)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1        
        self.a_s_cum = self.a_s_cum.to(self.device)
        self.sigmas_cum = self.sigmas_cum.to(self.device)
        self.a_s_prev = self.a_s_prev.to(self.device)
    
    def _pos_coeff(self):
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=self.device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
    
    def sample(self, x0, t, noise=None, noise_factor=1.):
        if noise is None:
          noise = noise_factor * torch.randn_like(x0)
        x_t = self.extract(self.a_s_cum, t, x0.shape) * x0 + \
                self.extract(self.sigmas_cum, t, x0.shape) * noise
        return x_t
    
    def pos_sample(self, x0, x_tp1, t, noise=None, noise_factor=1.):
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        if noise is None:
            noise = noise_factor * torch.randn_like(x0)
        
        mean = (self.extract(self.posterior_mean_coef1, t, x0.shape) * x0
                + self.extract(self.posterior_mean_coef2, t, x0.shape) * x_tp1)
        var = self.extract(self.posterior_variance, t, x0.shape)
        log_var = self.extract(self.posterior_log_variance_clipped, t, x0.shape)
        
        sample_x_pos = mean + nonzero_mask[:,None] * torch.exp(0.5 * log_var) * noise
        return sample_x_pos
    
    def fidelity(self, x, t, x_t_plus_one):
        b = x.shape[0]
        if self.posterior:
            mean = self.extract(self.a_s, t+1, x.shape) * x
            std = self.extract(self.sigmas, t+1, x.shape)
        else:
            mean = self.sample(x, t+1, noise=0)
            std = self.extract(self.sigmas_cum, t+1, x.shape)
            
        return (((x_t_plus_one - mean)/std)**2).view(b, -1).sum(1) / 2
    
    def init_sample(self, num):
        return torch.randn(num, 2).to(self.device)


class NoiseSampler(BaseSampler):
    def __init__(self, args, device):
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.discretize = args.discrete_type
        super().__init__(args, device)
        
        
    def _coeff(self):
        t = np.arange(0, self.T + 1, dtype=np.float64)
        t = t / self.T
        eps_small = 1e-3
        t = torch.from_numpy(t) * (1. - eps_small) + eps_small
        
        if self.discretize == 'geometric':
            var = var_func_geometric(t, self.beta_min, self.beta_max)
        elif self.discretize == 'square':
            var = var_func_vp(t**2, self.beta_min, self.beta_max)
        else:
            var = var_func_vp(t, self.beta_min, self.beta_max)
        
        alpha_bars = 1.0 - var
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        first = torch.tensor(1e-8)
        betas = torch.cat((first[None], betas)).to(self.device)
        self.betas = betas = betas.type(torch.float32)
        self.sigmas = betas**0.5
        self.a_s = torch.sqrt(1-betas)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1        
        self.a_s_cum = self.a_s_cum.to(self.device)
        self.sigmas_cum = self.sigmas_cum.to(self.device)
        self.a_s_prev = self.a_s_prev.to(self.device)
    
    def _pos_coeff(self):
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=self.device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
    
    def sample(self, x0, t, noise=None, noise_factor=1.):
        if noise is None:
          noise = noise_factor * torch.randn_like(x0)
        x_t = self.extract(self.a_s_cum, t, x0.shape) * x0 + \
                self.extract(self.sigmas_cum, t, x0.shape) * noise
        return x_t
    
    def pos_sample(self, x0, x_tp1, t, noise=None, noise_factor=1.):
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        if noise is None:
            noise = noise_factor * torch.randn_like(x0)
        
        mean = (self.extract(self.posterior_mean_coef1, t, x0.shape) * x0
                + self.extract(self.posterior_mean_coef2, t, x0.shape) * x_tp1)
        var = self.extract(self.posterior_variance, t, x0.shape)
        log_var = self.extract(self.posterior_log_variance_clipped, t, x0.shape)
        
        sample_x_pos = mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
        return sample_x_pos
    
    def fidelity(self, x, t, x_t_plus_one):
        b = x.shape[0]
        if self.posterior:
            mean = self.extract(self.a_s, t+1, x.shape) * x
            std = self.extract(self.sigmas, t+1, x.shape)
        else:
            mean = self.sample(x, t+1, noise=0)
            std = self.extract(self.sigmas_cum, t+1, x.shape)
            
        return (((x_t_plus_one - mean)/std)**2).view(b, -1).sum(1) / 2
    
    def init_sample(self, num):
        return torch.randn(num, 
                        self.num_channels, 
                        self.image_size, 
                        self.image_size).to(self.device)


class SRSampler(BaseSampler):
    def __init__(self, args, device):
        self.sf = args.sf
        self.scale = int(1/self.sf)
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.discretize = args.discrete_type
        super().__init__(args, device)
        
    def _coeff(self):
        n_timesteps = (self.T+1)//2
        t = np.arange(0, n_timesteps + 1, dtype=np.float64)
        t = t / n_timesteps
        eps_small = 1e-3
        t = torch.from_numpy(t) * (1. - eps_small) + eps_small
        
        if self.discretize == 'geometric':
            var = var_func_geometric(t, self.beta_min, self.beta_max)
        elif self.discretize == 'square':
            var = var_func_vp(t**2, self.beta_min, self.beta_max)
        else:
            var = var_func_vp(t, self.beta_min, self.beta_max)
        
        alpha_bars = 1.0 - var
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        first = torch.tensor(1e-8)
        betas = torch.cat((first[None], betas)).to(self.device)
        betas = betas.type(torch.float32)
        self.sigmas = betas**0.5
        self.a_s = torch.sqrt(1-betas)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(self.device)
        self.sigmas_cum = self.sigmas_cum.to(self.device)
        self.a_s_prev = self.a_s_prev.to(self.device)
        self.sigmas_sample_cum = self.scale**(torch.arange(len(self.sigmas_cum), device=self.device)) * self.sigmas_cum
    
    def sample(self, x0, t, noise=None, noise_factor=1.):
        if noise is None:
            noise = noise_factor * torch.randn_like(x0)
        x_t = self.extract(self.a_s_cum, torch.div(t+1, 2, rounding_mode='floor'), x0.shape) * x0\
            + self.extract(self.sigmas_sample_cum, torch.div(t+1, 2, rounding_mode='floor'), x0.shape) * noise
        x_t = downup(x_t, torch.div(t, 2, rounding_mode='floor'), x0.size(-1), self.sf)
        return x_t
    
    def fidelity(self, x, t, x_t_plus_one):
        b = x.shape[0]
        mean = self.sample(x, t+1, noise=0)
        std = self.extract(self.sigmas_cum, torch.div(t+2, 2, rounding_mode='floor'), x.shape)
        std = (1 - ((t[:,None,None,None]+1) % 2)*(1-self.scale)) * std
        return (((x_t_plus_one - mean) / std)**2).view(b, -1).sum(1) / 2
    
    def init_sample(self, num):
        img_size = int(self.image_size / ( (self.scale) ** torch.div(self.T, 2, rounding_mode='floor')))
        noise = 2 * torch.randn(num,
                                self.num_channels,
                                img_size,
                                img_size
                                ).to(self.device)
        return interp(noise, size=self.image_size, mode='nearest')


class VanillaSRSampler(BaseSampler):
    def __init__(self, args, device):
        """
        Create DDPM style Noise
        """
        self.sf = args.sf
        self.scale = int(1/self.sf)
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.discretize = args.discrete_type
        super().__init__(args, device)
        
    def _coeff(self):
        t = np.arange(0, self.T + 1, dtype=np.float64)
        t = t / self.T
        eps_small = 1e-3
        t = torch.from_numpy(t) * (1. - eps_small) + eps_small
        
        if self.discretize == 'geometric':
            var = var_func_geometric(t, self.beta_min, self.beta_max)
        elif self.discretize == 'square':
            var = var_func_vp(t**2, self.beta_min, self.beta_max)
        else:
            var = var_func_vp(t, self.beta_min, self.beta_max)
        
        alpha_bars = 1.0 - var
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        first = torch.tensor(1e-8)
        betas = torch.cat((first[None], betas)).to(self.device)
        betas = betas.type(torch.float32)
        self.sigmas = betas**0.5
        self.a_s = torch.sqrt(1-betas)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(self.device)
        self.sigmas_cum = self.sigmas_cum.to(self.device)
        self.a_s_prev = self.a_s_prev.to(self.device)
    
    def sample(self, x0, t, noise=None, noise_factor=1.):
        if noise is None:
            noise = noise_factor * torch.randn_like(x0)
        x_t = self.extract(self.a_s_cum, t, x0.shape) * x0\
        + self.extract(self.sigmas_cum, t, x0.shape) * noise
        x_t = downup(x_t, t, x0.size(-1), self.sf)
        return x_t
    
    def fidelity(self, x, t, x_t_plus_one):
        b = x.shape[0]
        mean = self.sample(x, t+1, noise=0)
        std = self.extract(self.sigmas_cum, t+1, x.shape)
        return (((x_t_plus_one - mean) / std)**2).view(b, -1).sum(1) / 2
    
    def init_sample(self, num):
        img_size = int(self.image_size / ( self.scale ** self.T ))
        noise = torch.randn(num,
                            self.num_channels,
                            img_size,
                            img_size
                            ).to(self.device)
        noise = noise / (self.scale ** self.T).float()
        return interp(noise, size=self.image_size, mode='nearest')


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def downup(x_t, t, size, sf):
    x = x_t.clone()
    for i in range(1, t.max()+1):
        x_t = interp(x_t, 
                scale_factor=sf,
                mode='bilinear',
                align_corners=False,
                recompute_scale_factor=True
                )
        x[t == i] = interp(x_t[t == i], 
                        size=size,
                        mode='nearest',
                        )
    return x