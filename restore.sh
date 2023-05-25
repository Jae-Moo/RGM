# CelebA-HQ-256
python3 restore.py --PnP_algo drs --dataset celeba_256 --image_size 256 \
--num_channels_dae 64 --n_mlp 3 \
--ch_mult 1 1 2 2 4 4 --noise_factor 0.99 --num_timesteps 4 \
--batch_size 100 --forward_name ddpm \
--ckpt train_logs/celeba_256/celeba/netG.pth \
--deg sr4 \
--alpha 0.05 --lmbda 10 --max_iter 40 --T 1

python3 restore.py --PnP_algo drs --dataset celeba_256 --image_size 256 \
--ch_mult 1 1 2 2 4 4 --noise_factor 0.99 --num_timesteps 4 \
--num_channels_dae 64 --n_mlp 3 \
--batch_size 100 --forward_name ddpm \
--ckpt train_logs/celeba_256/celeba/netG.pth \
--deg sr8 \
--alpha 0.05 --lmbda 10 --max_iter 40 --T 1 

python3 restore.py --PnP_algo drs --dataset celeba_256 --image_size 256 \
--ch_mult 1 1 2 2 4 4 --noise_factor 0.99 --num_timesteps 4 \
--num_channels_dae 64 --n_mlp 3 \
--batch_size 100 --forward_name ddpm \
--ckpt train_logs/celeba_256/celeba/netG.pth \
--deg color \
--alpha 0.5 --lmbda 5 --max_iter 20 --T 2 
