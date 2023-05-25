import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datetime import datetime
from forward import get_sampler
from utils import *


SYNTHETIC_EXP = ['gmm', 'checkerboard', 'gaussian25']

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:0')
    batch_size = args.batch_size
    nz = args.nz
    
    
    # Set Generator
    if args.dataset in SYNTHETIC_EXP:
        from models.toy import ToyGenerator
        netG = ToyGenerator(nz=nz).to(device)
    else:
        from models.ncsnpp_generator_adagn import NCSNpp
        netG = NCSNpp(args).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.schedule, eta_min=1e-5)
    netG = nn.DataParallel(netG)
    
    
    # Set Kernel
    if args.fixed_kernel:
        netD = lambda x,t:(None,x)
    else:
        if args.dataset in SYNTHETIC_EXP:
            from models.toy import ToyDiscriminator
            netD = ToyDiscriminator().to(device)
        elif args.dataset in ['mnist','cifar10']:
            from models.discriminator import Discriminator_small
            netD = Discriminator_small(nc = args.num_channels, ngf = args.ngf, t_emb_dim = args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
        else:
            from models.discriminator import Discriminator_large
            netD = Discriminator_large(nc = args.num_channels, ngf = args.ngf, t_emb_dim = args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
        
        optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.schedule, eta_min=1e-5)
        netD = nn.DataParallel(netD)
    
    
    # Set TransformNet if prior is dswd
    if args.prior_name == 'dswd':
        from models.discriminator import TransformNet
        if args.dataset in SYNTHETIC_EXP:
            netT = TransformNet(2).to(device)
        else:
            netT = TransformNet(args.ngf * 8 * 4 * 4).to(device)
        
        optimizerT = optim.Adam(netT.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
        netT = nn.DataParallel(netT)
    
    
    # Get Path
    exp = args.exp
    parent_dir = "./train_logs/{}".format(args.dataset)
    exp_path = os.path.join(parent_dir, exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    
    # Get Data
    data_loader = get_dataloader(args)
    
    
    # Resume
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    
    # Get sampling function
    sampler = get_sampler(args, device)
    
    
    # log file
    with open(os.path.join(exp_path, 'log.txt'), 'w') as f:
        f.write("Start Training")
        f.write('\n')
    
    
    # Start training
    start = datetime.now()
    for epoch in range(init_epoch, args.num_epoch+1):
        for _, x in enumerate(data_loader):
            try: x,_ = x
            except: pass
            
            real_data = x.float().to(device, non_blocking=True)
            
            if not args.fixed_kernel:
                for p in netD.parameters():  
                    p.requires_grad = True
                    
                netD.zero_grad()
                
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

                # real D loss
                x_t, x_tp1 = sampler.sample_pairs(real_data, t)
                x_t.requires_grad = True
                D_real, _ = netD(x_t, t)
                D_real = D_real.view(-1)
                
                errD_real = F.softplus(-D_real)
                errD_real = errD_real.mean()
                errD_real.backward(retain_graph=True)
                
                # lazy reg
                if args.lazy_reg is None:
                    grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()
                else:
                    if global_step % args.lazy_reg == 0:
                        grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                        grad_penalty = args.r1_gamma / 2 * grad_penalty
                        grad_penalty.backward()
                
                # fake D loss
                latent_z = torch.randn(batch_size, nz, device=device)

                x_0_predict = netG(x_tp1.detach(), t, latent_z)
                if args.posterior:
                    x_pos_sample = sampler.pos_sample(x_0_predict, x_tp1, t)
                else:
                    x_pos_sample = sampler.sample(x_0_predict.detach(), t)                
                output, _ = netD(x_pos_sample, t)
                output = output.view(-1)
                
                errD_fake = F.softplus(output)
                errD_fake = errD_fake.mean()
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()
            
                for p in netD.parameters():
                    p.requires_grad = False
            
            # Generator loss
            netG.zero_grad()
            
            latent_z = torch.randn(batch_size, nz, device=device)

            if args.prior_name == 'dswd':
                t = torch.randint(0, args.num_timesteps, (1,), device=device).repeat(real_data.size(0))                
            else:
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = sampler.sample_pairs(real_data, t)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            
            if args.posterior:
                x_pos_sample = sampler.pos_sample(x_0_predict, x_tp1, t)
            else:
                x_pos_sample = sampler.sample(x_0_predict, t)
                
            ## different loss for each prior
            if args.prior_name == 'gan':
                output, _ = netD(x_pos_sample, t)
                output = output.view(-1)
                errG = F.softplus(-output) - F.softplus(output)
                
            elif args.prior_name == 'dswd':
                _, features_fake = netD(x_pos_sample, t)
                _, features_real = netD(x_t, t)
                errG = DSWD(features_real.view(real_data.size(0),-1), features_fake.view(real_data.size(0),-1), args.num_projections, netT, optimizerT)
            
            elif args.prior_name == 'mmd':
                _, features_fake = netD(x_pos_sample, t)
                _, features_real = netD(x_t, t)
                features_fake = features_fake.view(real_data.size(0),-1)
                features_real = features_real.view(real_data.size(0),-1)
                errG = torch.sqrt(F.relu(MMD(features_real, features_fake, args.sigma_list)))
            
            ## fidelity loss
            if args.posterior:
                errReg = sampler.fidelity(x_pos_sample, t, x_tp1)
            else:
                errReg = sampler.fidelity(x_0_predict, t, x_tp1)
            
            err = errG + args.lmbda * errReg
            err = err.mean()
            err.backward()
            optimizerG.step()
            global_step += 1
            
            ## save losses
            if global_step % args.print_every == 0:
                with open(os.path.join(exp_path, 'log.txt'), 'a') as f:
                    f.write(f'Epoch {epoch:04d} : G Loss {errG.mean().item():.4f}, Reg Loss {errReg.mean().item():.4f}, Elapsed {datetime.now() - start}')
                    f.write('\n')
        
        if not args.no_lr_decay:
            schedulerG.step()
            if not args.fixed_kernel:
                schedulerD.step()
        
        # save content
        if epoch % args.save_content_every == 0:
            print('Saving content.')
            if args.fixed_kernel:
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                            'schedulerG': schedulerG.state_dict()}            
            else:
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                            'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                            'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            
            torch.save(content, os.path.join(exp_path, 'content.pth'))
        
        # save checkpoint
        if epoch % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


        # save generated images
        if epoch % args.save_image_every == 0:
            images = sampler.sample_from_model(generator=netG, num=batch_size)
            if args.is_image:
                images = (0.5*(images+1)).detach().cpu()
                torchvision.utils.save_image(images, os.path.join(exp_path, 'epoch_{}.png'.format(epoch)))
            else:
                images = images.detach().cpu().numpy()
                plt.plot(images[:,0], images[:,1], 'x')
                plt.savefig(os.path.join(exp_path, 'epoch_{}.png'.format(epoch)))
                plt.xlim(-3,3)
                plt.ylim(-3,3)
                plt.close()
                
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RGM parameters')
    
    # Experiment description
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--exp', default='d', help='name of experiment')
    parser.add_argument('--resume', action='store_true',default=False, help='Resume training or not')
    parser.add_argument('--dataset', default='cifar10', choices=['gmm', 'checkerboard', 'gaussian25','mnist', 'cifar10', 'lsun', 'celeba_256'], help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32, help='size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    
    # Generator configurations
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
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
    parser.add_argument('--not_use_tanh', action='store_true', default=False, help='use tanh for last layer')
    parser.add_argument('--z_emb_dim', type=int, default=256, help='embedding dimension of z')
    parser.add_argument('--t_emb_dim', type=int, default=256, help='embedding dimension of t')
    parser.add_argument('--nz', type=int, default=100, help='latent dimension')
    parser.add_argument('--ngf', type=int, default=64, help='The default number of channels of model')
    
    # Training/Optimizer configurations
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=900, help='the number of epochs')
    parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.0e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--schedule', type=int, default=1800, help='lr scheduler, learning rate 1e-5 until {schedule} epoch')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    parser.add_argument('--r1_gamma', type=float, default=0.02, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=15, help='lazy regulariation.')
    parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    # Fidelity term configurations
    parser.add_argument('--forward_name', type=str, default='ddpm', choices=['toy', 'ddpm', 'sr'], help='degradation name')
    parser.add_argument('--discrete_type', type=str, default='original', choices=['original', 'square', 'geometric'], help='The type of std (of noise)')
    parser.add_argument('--num_timesteps', type=int, default=4, help='The number of timesteps')
    parser.add_argument('--posterior', action='store_true', default=False, help='Use posterior sampling')
    parser.add_argument('--lmbda', type=float, default=0.001, help='scale of fidelity loss')
    parser.add_argument('--beta_min', type=float, default=0.1, help='minimum beta for noise')
    parser.add_argument('--beta_max', type=float, default=20.0, help='maximum beta for noise')
    parser.add_argument('--sf', type=float, default=0.5, help='scale factor used for SR')

    # Prior term configurations
    parser.add_argument('--prior_name', type=str, default='gan', choices=['gan', 'dswd', 'mmd'], help='type of prior')
    parser.add_argument('--fixed_kernel', action='store_true', default=False, help='If fixed_kernel, we do not use discriminator')
    parser.add_argument('--num_projections', type=int, default=1000, help='The number of projections for dswd')
    parser.add_argument('--sigma_list', nargs='+', type=float, default=[1,2,4,8,16], help='sigmas used for MMD distance')
        
    # Visualize/Save configurations
    parser.add_argument('--print_every', type=int, default=100, help='print current loss for every x iterations')
    parser.add_argument('--save_content_every', type=int, default=5, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=100, help='save ckpt every x epochs')
    parser.add_argument('--save_image_every', type=int, default=10, help='save images every x epochs')
    
    args = parser.parse_args()
    train(args)