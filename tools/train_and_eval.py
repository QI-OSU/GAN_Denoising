import torch, os, cv2
import numpy as np
from tqdm import tqdm
import dataset.utils as util


def train_one_epoch(model, lpips_model, discriminator, data_loader, optimizer_g, optimizer_d, step_count,
                    recon_criterion, disc_criterion, phs_criterion, disc_step_start, acc_steps,
                    train_config, epoch_idx, device):
    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    recon_loss_lst, perceptual_loss_lst, phs_loss_lst, disc_loss_lst, gen_loss_lst = [], [], [], [], []
    for idx, data in tqdm(enumerate(data_loader), desc=f"Epoch {epoch_idx}", total=len(data_loader)):
        if idx > 10: break
        step_count += 1
        im = data['dint'].to(device)
        lab = data['lab'].to(device)

        # Fetch autoencoders output(reconstructions)
        output = model(im)
        x_diff = util.getDifference(output, lab)
        x_lab = torch.zeros_like(x_diff)
        # Fetch phs loss
        phs_loss = phs_criterion(output)
        phs_loss_lst.append(phs_loss.item())
        phs_loss = phs_loss / acc_steps
        g_loss = phs_loss

        ######### Optimize Generator ##########
        # L2 Loss
        recon_loss = recon_criterion(x_diff, x_lab)
        recon_loss_lst.append(recon_loss.item())
        recon_loss = recon_loss / acc_steps
        g_loss += recon_loss

        # Adversarial loss only if disc_step_start steps passed
        if step_count > disc_step_start:
            disc_fake_pred = discriminator(output)
            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.ones(disc_fake_pred.shape,
                                                       device=disc_fake_pred.device))
            gen_loss_lst.append(train_config['disc_weight'] * disc_fake_loss.item())
            g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
        lpips_loss = torch.mean(lpips_model(output, lab)) / acc_steps
        perceptual_loss_lst.append(train_config['perceptual_weight'] * lpips_loss.item())
        g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps
        g_loss.backward()
        #####################################

        ######### Optimize Discriminator #######
        if step_count > disc_step_start:
            fake = output
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(lab)
            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.zeros(disc_fake_pred.shape,
                                                        device=disc_fake_pred.device))
            disc_real_loss = disc_criterion(disc_real_pred,
                                            torch.ones(disc_real_pred.shape,
                                                       device=disc_real_pred.device))
            disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
            disc_loss_lst.append(disc_loss.item())
            disc_loss = disc_loss / acc_steps
            disc_loss.backward()
            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()
        #####################################

        if step_count % acc_steps == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()

    optimizer_d.step()
    optimizer_d.zero_grad()
    optimizer_g.step()
    optimizer_g.zero_grad()

    if step_count > disc_step_start:
        print(
            'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
            'Phs loss : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
            format(epoch_idx,
                   np.mean(recon_loss_lst),
                   np.mean(perceptual_loss_lst),
                   np.mean(phs_loss_lst),
                   np.mean(gen_loss_lst),
                   np.mean(disc_loss_lst)))

        return {'recon_loss': np.mean(recon_loss_lst),
                'perceptual_loss': np.mean(perceptual_loss_lst),
                'phs_loss': np.mean(phs_loss_lst),
                'gen_loss': np.mean(gen_loss_lst),
                'disc_loss': np.mean(disc_loss_lst),
                'step_count': step_count}, step_count

    else:
        print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Phs loss : {:.4f}'.
              format(epoch_idx,
                     np.mean(recon_loss_lst),
                     np.mean(perceptual_loss_lst),
                     np.mean(phs_loss_lst)))

        return {'recon_loss': np.mean(recon_loss_lst),
                'perceptual_loss': np.mean(perceptual_loss_lst),
                'phs_loss': np.mean(phs_loss_lst),
                'step_count': step_count}, step_count


def evaluate(model, data_loader, epoch, train_config, device):
    model.eval()
    sample_num = 5

    tif_save_path = f'{train_config["tif_save_path"]}/{epoch}'
    png_save_path = f'{train_config["png_save_path"]}/{epoch}'

    os.makedirs(tif_save_path, exist_ok=True)
    os.makedirs(png_save_path, exist_ok=True)

    data_loader_iter = iter(data_loader)

    with torch.no_grad():
        for idx in tqdm(range(sample_num), desc=f'Evaluate on epoch: {epoch}'):
            data = data_loader_iter.next()

            im = data['dint'].to(device)
            lab = data['lab'].to(device)
            output = model(im)
            diff = util.getDifference(lab, output)

            # tensor2numpy
            pred = util.tensor2img(output)
            lab = util.tensor2img(lab)
            dint = util.tensor2img(im)
            diff = util.tensor2img(diff)

            # save tif
            util.save_tif(pred, os.path.join(tif_save_path, f'{idx + 1}_pred.tif'))
            util.save_tif(lab, os.path.join(tif_save_path, f'{idx + 1}_lab.tif'))
            util.save_tif(dint, os.path.join(tif_save_path, f'{idx + 1}_dint.tif'))
            util.save_tif(diff, os.path.join(tif_save_path, f'{idx + 1}_diff.tif'))

            # save png
            util.save_png(pred, os.path.join(png_save_path, f'{idx + 1}_pred.png'))
            util.save_png(lab, os.path.join(png_save_path, f'{idx + 1}_lab.png'))
            util.save_png(dint, os.path.join(png_save_path, f'{idx + 1}_dint.png'))
            util.save_png(diff, os.path.join(png_save_path, f'{idx + 1}_diff.png'))

    model.train()


