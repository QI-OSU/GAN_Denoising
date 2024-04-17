import yaml
import argparse
import torch
import os
import numpy as np
# from models.vqvae import VQVAE
# from models.unet_base import Unet
from models.unet import UNet
from models.unet_simple import create_model
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_dataset import MnistDataset
from torch.optim import Adam
from dataset.utils import save_loss, load_loss, plot_loss, print_gpu_info, PhaseContinuityLoss
from train_and_eval import train_one_epoch, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_gpu_info()


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    # autoencoder_config = config['autoencoder_params']
    # diffusion_config = config['ldm_params']
    unet_config = config['unet_params']
    train_config = config['train_params']

    # Create the model and dataset #
    # model = UNet(in_channel=unet_config['in_channel'],
    #             out_channel=unet_config['out_channel'],
    #             inner_channel=unet_config['inner_channel'],
    #             channel_mults=unet_config['channel_multiplier'],
    #             attn_res=unet_config['attn_res'],
    #             res_blocks=unet_config['res_blocks'],
    #             dropout=unet_config['dropout'],
    #             image_size=unet_config['image_size']).to(device)

    model = create_model(num_classes=1).to(device)

    # Create the dataset
    im_dataset = MnistDataset(split='train',
                              im_root=dataset_config['im_path'],
                              noise_root=dataset_config['noise_path'],
                              im_size=dataset_config['im_size'],
                              im_channels=dataset_config['im_channels'])

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    val_data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=True)

    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    phs_criterion = PhaseContinuityLoss()

    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS(weight_path=train_config['vgg16_pretrained']).eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)

    model_without_ddp = model
    lpips_model_without_ddp = lpips_model
    discriminator_without_ddp = discriminator

    if train_config['distributed']:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model_without_ddp = model.module

        lpips_model = torch.nn.DataParallel(lpips_model, device_ids=[0, 1, 2, 3])
        lpips_model_without_ddp = lpips_model.module

        discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1, 2, 3])
        discriminator_without_ddp = discriminator.module

    optimizer_d = Adam(discriminator_without_ddp.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model_without_ddp.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    disc_step_start = train_config['disc_start']
    step_count, resume_epoch = 1, 1

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    (epoch_lst, epoch_lst_ad, recon_loss_epoch, perceptual_loss_epoch, phs_loss_epoch,
     disc_loss_epoch, gen_loss_epoch) = [], [], [], [], [], [], []

    if train_config['resume']:
        checkpoint = torch.load(train_config['resume'], map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        discriminator_without_ddp.load_state_dict(checkpoint['discriminator'])
        lpips_model_without_ddp.load_state_dict(checkpoint['lpips_model'])

        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        resume_epoch = checkpoint['epoch'] + 1
        step_count = checkpoint['step_count']

        recon_loss_dict = load_loss(os.path.join(train_config['loss_save_path'], 'recon_loss_epoch.json'))
        perceptual_loss_dict = load_loss(os.path.join(train_config['loss_save_path'], 'perceptual_loss_epoch.json'))
        phs_loss_dict = load_loss(os.path.join(train_config['loss_save_path'], 'phs_loss_epoch.json'))
        gen_loss_dict = load_loss(os.path.join(train_config['loss_save_path'], 'gen_loss_epoch.json'))
        disc_loss_dict = load_loss(os.path.join(train_config['loss_save_path'], 'disc_loss_epoch.json'))

        recon_loss_epoch = recon_loss_dict['loss'][:recon_loss_dict['epoch'].index(resume_epoch)]
        perceptual_loss_epoch = perceptual_loss_dict['loss'][:perceptual_loss_dict['epoch'].index(resume_epoch)]
        phs_loss_epoch = phs_loss_dict['loss'][:phs_loss_dict['epoch'].index(resume_epoch)]
        gen_loss_epoch = gen_loss_dict['loss'][:gen_loss_dict['epoch'].index(resume_epoch)]
        disc_loss_epoch = disc_loss_dict['loss'][:disc_loss_dict['epoch'].index(resume_epoch)]
        epoch_lst = recon_loss_dict['epoch'][:recon_loss_dict['epoch'].index(resume_epoch)]
        epoch_lst_ad = gen_loss_dict['epoch'][:gen_loss_dict['epoch'].index(resume_epoch)]  # 1 2 3 4 5 6 7 8

        print(f'---resume from epoch: {resume_epoch}---')

    for epoch_idx in range(resume_epoch, resume_epoch + num_epochs):
        print(step_count)
        loss_dict, step_count = train_one_epoch(model, lpips_model, discriminator, data_loader,
                                    optimizer_g, optimizer_d, step_count,
                                    recon_criterion, disc_criterion, phs_criterion,
                                    disc_step_start, acc_steps, train_config,
                                    epoch_idx, device)
        recon_loss_epoch.append(loss_dict['recon_loss'].item())
        perceptual_loss_epoch.append(loss_dict['perceptual_loss'].item())
        phs_loss_epoch.append(loss_dict['phs_loss'].item())
        loss_lst = ['recon_loss_epoch', 'perceptual_loss_epoch', 'phs_loss_epoch']
        loss_lst_ad = ['gen_loss_epoch', 'disc_loss_epoch']

        if loss_dict.get('gen_loss') and loss_dict.get('disc_loss'):
            gen_loss_epoch.append(loss_dict['gen_loss'].item())
            disc_loss_epoch.append(loss_dict['disc_loss'].item())
            epoch_lst_ad.append(epoch_idx)

        epoch_lst.append(epoch_idx)

        os.makedirs(train_config['loss_save_path'], exist_ok=True)

        for loss in loss_lst:
            save_loss(eval(loss), epoch_lst, os.path.join(train_config['loss_save_path'], loss + '.json'))
            plot_loss(eval(loss), epoch_lst, train_config['loss_save_path'], loss_name=loss)

        if epoch_lst_ad:
            for loss in loss_lst_ad:
                save_loss(eval(loss), epoch_lst_ad, os.path.join(train_config['loss_save_path'], loss + '.json'))
                plot_loss(eval(loss), epoch_lst_ad, train_config['loss_save_path'], loss_name=loss)

        weight_save_path = os.path.join(os.getcwd(), train_config['weight_save_path'])
        os.makedirs(weight_save_path, exist_ok=True)

        if epoch_idx % train_config['validate_interval'] == 0:
            evaluate(model, val_data_loader, epoch_idx, train_config, device)

            save_file = {"model": model_without_ddp.state_dict(),
                         "lpips_model": lpips_model_without_ddp.state_dict(),
                         "discriminator": discriminator_without_ddp.state_dict(),
                         "optimizer_d": optimizer_d.state_dict(),
                         "optimizer_g": optimizer_g.state_dict(),
                         "epoch": epoch_idx,
                         "step_count": loss_dict['step_count'],
                         }
            torch.save(save_file, f"{weight_save_path}/model_{epoch_idx}.pth")

    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
