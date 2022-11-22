# Structure-Preserving GANs

Tensorflow 2 implementation of *Structure-preserving GANs* presented at ICML 2022


## Dependencies

-tensorflow-gpu=2.7.0
-tensorflow-addons==0.11.2
-gcnn implemented by Neel Dey
 git+https://github.com/neel-dey/tf2-GrouPy#egg=GrouPy -e git+https://github.com/neel-dey/tf2-keras-gcnn.git#egg=keras_gcnn

## Usage: run the following command to reproduce the results in the paper

## Example: RotMNIST using RA-GANs with 10% training data

#1 CNN G&D
python train_script_rel.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch z2_rotmnist --d_arch z2_rotmnist --d_updates 2 --dataset rotmnist --num_samples 6000

#2 Eqv G + CNN D
python train_script_rel.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch correct_p4_rotmnist --d_arch z2_rotmnist --d_updates 2 --dataset rotmnist --num_samples 6000

#3 CNN G + Inv D
python train_script_rel.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch z2_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --num_samples 6000

#4 (w)Eqv G + Inv D
python train_script_rel.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch p4_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --num_samples 6000

#5 Eqv G + Inv D
python train_script_rel.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch correct_p4_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --num_samples 6000

#6 Eqv G + Inv D, 8 rotations
python train_script_rel.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch cutcorner_lvl1_se2_8_rotmnist --d_arch se2_8_rotmnist --d_updates 2 --dataset rotmnist --num_samples 6000


## Example: RotMNIST using Lip-alpha-GAN with 10% training data

#1 CNN G&D
python3 train_script_alpha_div.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch z2_rotmnist --d_arch z2_rotmnist --d_updates 2 --dataset rotmnist --alpha 2 --reverse 0 --num_samples 6000

#2 Eqv G + CNN D
python3 train_script_alpha_div.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch correct_p4_rotmnist --d_arch z2_rotmnist --d_updates 2 --dataset rotmnist --alpha 2 --reverse 0 --num_samples 6000

#3 CNN G + Inv D
python3 train_script_alpha_div.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch z2_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --alpha 2 --reverse 0 --num_samples 6000

#4 (w)Eqv G + Inv D
python3 train_script_alpha_div.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch p4_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --alpha 2 --reverse 0 --num_samples 6000

#5 Eqv G + Inv D
python3 train_script_alpha_div.py --epochs 660 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch correct_p4_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --alpha 2 --reverse 0 --num_samples 6000

#6 Eqv G + Inv D, 8 rotations
python3 train_script_alpha_div.py --epochs 6600 --batchsize 64 --name samplerotmnist --latent_dim 64 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch cutcorner_lvl1_se2_8_rotmnist --d_arch p4_rotmnist --d_updates 2 --dataset rotmnist --alpha 2 --reverse 0 --num_samples 6000

## Example: ANHIR using RA-GANs

#1 CNN G&D
python train_script_rel.py --epochs 136 --batchsize 32 --name sampleanhir64 --latent_dim 128 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch z2_anhir64 --d_arch z2_anhir64 --d_updates 2 --dataset anhir64

#2 (W)Eqv G + Inv D
python train_script_rel.py --epochs 136 --batchsize 32 --name sampleanhir64 --latent_dim 128 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch p4m_anhir64 --d_arch p4m_anhir64 --d_updates 2 --dataset anhir64

#3 Eqv G + Inv D
python train_script_rel.py --epochs 136 --batchsize 32 --name sampleanhir64 --latent_dim 128 --gp_wt 0.1 --lr_g 1e-4 --lr_d 4e-4 --g_arch correct_p4m_anhir64 --d_arch p4m_anhir64 --d_updates 2 --dataset anhir64


## Example: LYSTO using Lip-alpha-GAN

#1 CNN G&D
python train_script_alpha_div.py --epochs 161 --batchsize 32 --name samplelysto64 --latent_dim 128 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch z2_lysto64 --d_arch z2_lysto64 --d_updates 2 --dataset lysto64  --alpha 2.0 --reverse 0

#2 (W)Eqv G + Inv D
python train_script_alpha_div.py --epochs 161 --batchsize 32 --name samplelysto64 --latent_dim 128 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch p4m_lysto64 --d_arch p4m_lysto64 --d_updates 2 --dataset lysto64  --alpha 2.0 --reverse 0

#3 Eqv G + Inv D
python train_script_alpha_div.py --epochs 161 --batchsize 32 --name samplelysto64 --latent_dim 128 --gp_wt 10.0 --lr_g 1e-4 --lr_d 4e-4 --g_arch correct_p4m_lysto64 --d_arch p4m_lysto64 --d_updates 2 --dataset lysto64  --alpha 2.0 --reverse 0
