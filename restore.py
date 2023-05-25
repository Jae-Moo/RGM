import argparse
import torch
import numpy as np
import os
import torchvision
import math
from tqdm import tqdm
from utils import get_testloader
# sampler
from forward import get_sampler
from torch.autograd import grad
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from utils import get_model


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))
    
    def calculate_prox(self, lmbda, x, trans_y):
        singulars = self.singulars()
        temp = self.Vt(x) + self.Vt(lmbda * trans_y)
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / (1 + lmbda * singulars**2)
        return self.V(temp)
    
    def calculate_grad(self, lmbda, x, trans_y):
        singulars = self.singulars()
        temp = self.Vt(x)
        temp = temp[:, :singulars.shape[0]] * (lmbda * singulars**2)
        temp = self.V(self.add_zeros(temp))
        temp = temp - lmbda * trans_y
        return temp


# Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


# Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


# Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        #Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WH, C'
        #multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WH, C
        #permute back to vector representation
        recon = needles.permute(0, 2, 1) #shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WH, C
        #multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WH, C'
        #reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, :self.img_dim**2] = reshaped
        return temp


# Degrade image / calculate Proximal
class RestoreCalculator:
    def __init__(self, args):
        self.deg = deg = args.deg
        self.H_funcs = None
        self.device = 'cuda'
        self.shape = (args.num_channels, args.image_size, args.image_size)
        
        if deg == 'deno':
            self.H_funcs = Denoising(args.num_channels, args.image_size, self.device)
        
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            self.H_funcs = SuperResolution(args.num_channels, args.image_size, blur_by, self.device)
        
        elif deg == 'color':
            self.H_funcs = Colorization(args.image_size, self.device)
        
        else:
            print("ERROR: degradation type not supported")
            quit()
        
        self.sigma_0 = 2 * args.sigma_0 / 255. #to account for scaling to [-1,1]

    def vec2img(self, y_0):
        pinv_y_0 = self.H_funcs.H_pinv(y_0).view(y_0.shape[0], *self.shape)
        if self.deg[:6] == 'deblur': pinv_y_0 = y_0.view(y_0.shape[0], *self.shape)
        elif self.deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, *self.shape[1:]).repeat(1, 3, 1, 1)
        elif self.deg[:3] == 'inp': pinv_y_0 += self.H_funcs.H_pinv(self.H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1
        return pinv_y_0

    def degrade(self, original_x, noise=None):
        degraded_img = self.H_funcs.H(original_x)
        if noise is None:
            noise = self.sigma_0 * torch.randn_like(degraded_img, device=self.device) 
        degraded_img = degraded_img + noise
        return degraded_img
    
    def initialize_prox(self, y):
        return self.H_funcs.Ht(y)
    
    def loss(self, x, degraded_img_vec, degraded_img):
        if 'inp' in self.deg:
            return (self.mask*(x - degraded_img)**2).view(x.shape[0], -1).mean(1)
        else:
            return ((self.degrade(x, 0) - degraded_img_vec)**2).view(x.shape[0], -1).mean(1)
    
    def calculate_prox(self, lmbda, x, y_trans):
        return self.H_funcs.calculate_prox(lmbda, x, y_trans).view(x.shape[0], *self.shape)
    
    def calculate_grad(self, lmbda, x, y_trans):
        return self.H_funcs.calculate_grad(lmbda, x, y_trans).view(x.shape[0], *self.shape)
    
    def psnr(self, x_restored, x_original):
        assert x_restored.shape == x_original.shape
        b = x_original.shape[0]
        mse = torch.mean(((x_restored-x_original)**2).view(b, -1), dim=1)
        return 20 * torch.log10(1/torch.sqrt(mse)).mean()

    def ssim(self, x_restored, x_original):
        return SSIM()(x_restored, x_original)


# Restoration module
class PnP_algorithm:
    def __init__(self, model, sampler, calculator, args):
        self.model = model
        self.calculator = calculator
        self.sampler = sampler
        self.args = args
        self.T = args.T
        self.max_iter = args.max_iter
        self.nz = args.nz
        self.noise_factor = args.noise_factor
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.algorithm = {'drs': self.DRS}
        
    def restore(self, degraded_img, degraded_img_vec, alg_name, clean_img=None):
        restored_img = self.algorithm[alg_name](degraded_img, degraded_img_vec, clean_img)
        return restored_img

    def DRS(self, degraded_img, degraded_img_vec, clean_img=None):
        # get Ht_y
        degraded_img_vec = degraded_img_vec.to(self.device)
        Ht_y = self.calculator.initialize_prox(degraded_img_vec)
        
        b = degraded_img.shape[0] # batch_size
        x = degraded_img
        with torch.no_grad():
            for i in range(self.max_iter):
                # Get t (how much to noise)
                for j in reversed(range(1,self.T+1)):
                    t = j * torch.ones(b, dtype=torch.int64, device=self.device)
                
                    # Get x_t
                    x_t = self.sampler.sample(x, t, noise_factor=self.noise_factor)
                    torchvision.utils.save_image(torch.clamp((x_t + 1.)*0.5 , 0, 1).cpu(), os.path.join(args.savepath, 'xt.png'))
                    
                    # Get Latent
                    latent_z = torch.randn((b, self.nz), device=self.device)
                
                    # Update y by moving average
                    x_update = self.model(x_t, t-1, latent_z)
                    y = (1 - self.alpha) * x + self.alpha * x_update
                
                    # Hard constraint (If restriction is needed)
                    if self.args.clamp:
                        y = torch.clamp(y, -1, 1)
                
                    # Go Proximal step
                    z = self.calculator.calculate_prox(self.lmbda, 2 * y - x, Ht_y)
                    x = x + (z - y)
        return x.cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Restoration parameters')
    parser.add_argument('--batch_size', type=int, default=100)
    
    # generator config
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,2], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--num_timesteps', type=int)
    
    # task
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--deg', type=str, choices=['deno', 'sr2', 'sr4', 'sr8', 'color'])
    parser.add_argument('--sigma_0', type=float, default=0.)
    parser.add_argument('--dataset', type=str, default='cifar10')
    
    # pretrained model & sampler path
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--forward_name', type=str, choices=['ddpm', 'vanilla_sr', 'sr'])
    parser.add_argument('--discrete_type', type=str, default='original', choices=['original', 'square', 'geometric'], help='The type of discretization')
    
    # pnp arguments    
    parser.add_argument('--PnP_algo', type=str, default='drs', choices=['drs'])
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--noise_factor', type=float, default=0.99)
    parser.add_argument('--clamp', action='store_true', default=False)
    
    # save path
    parser.add_argument('--savepath', type=str, default='restore_results')
    
    args = parser.parse_args()
    
    
    # create save path
    args.savepath = os.path.join(args.savepath, f'{args.dataset}_{args.deg}_{args.sigma_0}_{args.forward_name}')
    os.makedirs(args.savepath, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    sampler = get_sampler(args, device)
    calculator = RestoreCalculator(args)
    dataloader = get_testloader(args)
    
    
    for i, clean_img in enumerate(dataloader):
        try: clean_img,_=clean_img
        except:pass
        
        clean_img = clean_img.to(device)[:1]
        degraded_img_vec = calculator.degrade(clean_img)
        degraded_img = calculator.vec2img(degraded_img_vec)
    
    
    model = get_model(args, args.ckpt).to(device)
    pnp_module = PnP_algorithm(model, sampler, calculator, args)
    
    print('Successfully matched all modules.')
    
    # restore images
    AFTER_PSNR = []
    AFTER_SSIM = []
    
    for i, clean_img in enumerate(dataloader):
        try: clean_img, _ = clean_img
        except: pass
        
        clean_img = clean_img.to(device)
        degraded_img_vec = calculator.degrade(clean_img)
        degraded_img = calculator.vec2img(degraded_img_vec)
        
        if i == 0:
            torchvision.utils.save_image((clean_img + 1.) / 2., os.path.join(args.savepath, 'original.png'))
            torchvision.utils.save_image(torch.clamp((degraded_img + 1.) / 2., 0, 1), os.path.join(args.savepath, 'degraded.png'))
        
        restored_img = pnp_module.restore(degraded_img, degraded_img_vec, args.PnP_algo, clean_img)
        restored_img = (restored_img + 1) / 2
        restored_img = torch.clamp(restored_img, 0, 1)
        
        if i == 0:
            torchvision.utils.save_image(restored_img, os.path.join(args.savepath, 'restored.png'))
        
        before_psnr = calculator.psnr(torch.clamp((degraded_img + 1.) / 2., 0, 1).cpu(), ((clean_img + 1.) / 2.).cpu())
        after_psnr = calculator.psnr(restored_img, ((clean_img + 1.) / 2.).cpu())
        before_ssim = calculator.ssim(torch.clamp((degraded_img + 1.) / 2., 0, 1).cpu(), ((clean_img + 1.) / 2.).cpu())
        after_ssim = calculator.ssim(restored_img.cpu(), ((clean_img + 1.) / 2.).cpu())

        AFTER_PSNR.append(after_psnr)
        AFTER_SSIM.append(after_ssim)
    

    Total = sum(AFTER_PSNR)/len(AFTER_PSNR)
    print(f'After mean psnr : {Total.item()}')
    
    Total = sum(AFTER_SSIM)/len(AFTER_SSIM)
    print(f'After mean ssim : {Total.item()}')
    print('=====================================')