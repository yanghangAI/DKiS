#!/usr/bin/env python
import time
import torch
import torch.nn
import torch.optim
import numpy as np
import config as c
from torch.utils.tensorboard import SummaryWriter
import datasets
import viz
import warnings
import utils
from utils import attack, gauss_noise, mse_loss, computePSNR
import torchvision
import model as M
import random
import threading
import queue


def get_k(shape):
    # -1, -0.5, 0.5, 1
    k = torch.randn(shape).cuda()
    k[k <= 0] = -1
    k[k > 0] = 1
    k2_list = []
    for i in range(16):
        k2 = np.arange(16)
        np.random.shuffle(k2)
        k2_list.append(k2)
    k_list = []
    for i in range(16):
        k_list.append([k[i], k2_list[i]])
    return k_list


def put_k(shape, q):
    # -1, -0.5, 0.5, 1
    while True:
        q.put(get_k(shape))


def load(net, optim, name, load_opt=True):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items()}
    net.load_state_dict(network_state_dict)
    if load_opt == True:
        optim.load_state_dict(state_dicts['opt'])

    return net, optim


def embed_attack(net, input_img, attack_method, k):
    #################
    #    forward:   #
    #################
    output = net(input_img, k)
    output_container = output.narrow(1, 0, channels_in)
    container_img = iwt(output_container)
    output_z = output.narrow(1, channels_in, channels_in)
    #################
    #   attack:   #
    #################
    attack_container = attack(container_img, attack_method)

    return container_img, attack_container, output_container, output_z


def train_epoch(net, optim=None, attack_method=None, i_epoch=None, writer=None, mode='train', lam=(1.0, 1.0),
                device='cuda', iswrite=True, dataloader=None, k_queue=None):
    r_loss_list, g_loss_list, pre_loss_list, post_loss_list, psnr_c, psnr_s, total_loss_list = [], [], [], [], [], [], []
    psnr_s2, s2_loss_list, k_loss_list = [], [], []
    lam_c, lam_s = lam
    display = random.randint(0, 40)

    for i_batch, data in enumerate(dataloader):
        data = data.to(device)
        num = data.shape[0] // 2
        host = data[:num]
        secret = data[num:num * 2]
        host_input = dwt(host)
        secret_input = dwt(secret)
        input_img = torch.cat((host_input, secret_input), 1)
        k = k_queue.get()

        container, attack_container, output_container, output_z = embed_attack(net, input_img, attack_method, k)

        input_container = dwt(attack_container)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((input_container, gauss_noise(secret_input.shape)), 1)
        output_image = net(output_rev, k, rev=True)
        extracted = output_image.narrow(1, channels_in, channels_in)
        extracted = iwt(extracted)

        if mode == 'val':
            with torch.no_grad():
                k_ = k_queue.get()
                output_image_ = net(output_rev, k_, rev=True)
                extracted_ = output_image_.narrow(1, channels_in, channels_in)
                extracted_ = iwt(extracted_)

            s2_loss = mse_loss(extracted_, secret).item()
            extracted_ = extracted_.clip(0, 1)
            psnr_temp = computePSNR(extracted_, secret)
            psnr_s2.append(psnr_temp)
            s2_loss_list.append(s2_loss)
            if i_batch == display:
                s1_display = extracted[0].detach().cpu().clip(0, 1)
                s2_display = extracted_[0].detach().cpu().clip(0, 1)
                c_display = container[0].detach().cpu().clip(0, 1)

        #################
        #     loss:     #
        #################

        c_loss = mse_loss(container, host)
        s_loss = mse_loss(extracted, secret)

        total_loss = lam_c * c_loss + lam_s * s_loss

        if mode == 'train':
            total_loss.backward()
            optim.step()
            optim.zero_grad()


        elif mode == 'test':
            extracted = extracted.clip(0, 1)
            secret = secret.clip(0, 1)
            host = host.clip(0, 1)
            container = container.clip(0, 1)
            torchvision.utils.save_image(host, c.IMAGE_PATH_host + '%.5d.png' % i_batch)
            torchvision.utils.save_image(container, c.IMAGE_PATH_container + '%.5d.png' % i_batch)
            torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i_batch)
            torchvision.utils.save_image(extracted, c.IMAGE_PATH_extracted + '%.5d.png' % i_batch)

        total_loss_list.append(total_loss.item())

    if iswrite:
        extracted = extracted.clip(0, 1)
        secret = secret.clip(0, 1)
        host = host.clip(0, 1)
        container = container.clip(0, 1)

        psnr_temp = computePSNR(extracted, secret)
        psnr_s.append(psnr_temp)
        psnr_temp_c = computePSNR(host, container)
        psnr_c.append(psnr_temp_c)
        g_loss_list.append(c_loss.item())
        r_loss_list.append(s_loss.item())
        if mode == 'val':
            before = 'val_'
            writer.add_scalars("PSNR_S2", {"average psnr": np.mean(psnr_s2)}, i_epoch)
            writer.add_scalars("s2_loss", {"rev loss": np.mean(s2_loss_list)}, i_epoch)
            writer.add_image('s1', s1_display, i_epoch)
            writer.add_image('s2', s2_display, i_epoch)
            writer.add_image('c', c_display, i_epoch)
        else:
            before = ''
        writer.add_scalars(f"{before}c_loss", {f"{before}guide loss": np.mean(g_loss_list)}, i_epoch)
        writer.add_scalars(f"{before}s_loss", {f"{before}rev loss": np.mean(r_loss_list)}, i_epoch)
        writer.add_scalars(f"{before}PSNR_S", {f"{before}average psnr": np.mean(psnr_s)}, i_epoch)
        writer.add_scalars(f"{before}PSNR_C", {f"{before}average psnr": np.mean(psnr_c)}, i_epoch)
        writer.add_scalars(f"{before}Loss", {f"{before}Loss": np.mean(total_loss_list)}, i_epoch)

    return np.mean(total_loss_list)


def train(net, optim, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer=None, expinfo='',
          lam=(0.5, 0.5)):
    writer = SummaryWriter(comment=expinfo, filename_suffix="steg")
    net.eval()
    val_loss = train_epoch(net, optim, attack_method, start_epoch, mode='val', writer=writer, lam=lam, iswrite=False,
                           dataloader=testloader, k_queue=val_k_queue)
    net.train()
    tic = time.time()
    for i_epoch in range(start_epoch + 1, end_epoch + 1):
        if i_epoch % c.val_freq == 0:
            iswrite = True
        else:
            iswrite = False
        #################
        #     train:    #
        #################
        train_loss = train_epoch(net, optim, attack_method, i_epoch, writer=writer, lam=lam, iswrite=iswrite,
                                 dataloader=trainloader, k_queue=k_queue)
        toc = time.time()
        t = toc - tic
        tic = toc
        #################
        #      val:     #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                net.eval()
                val_loss = train_epoch(net, optim, attack_method, i_epoch, mode='val', writer=writer, lam=lam,
                                       iswrite=True, dataloader=testloader, k_queue=val_k_queue)
                net.train()
        info = [np.round(train_loss, 4), np.round(val_loss, 4), np.round(np.log10(optim.param_groups[0]['lr']), 4),
                attack_method, f'{np.round(t, 2)}s']

        viz.show_loss(visualizer, info)

        if i_epoch > 0 and (i_epoch % 25) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')

        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, f'final_state/{expinfo}.pt')
    writer.close()


def main(attack_method, start_epoch=0, end_epoch=1600, lam=(1.0, 1.0), exp_info=''):
    warnings.filterwarnings("ignore")
    alpha_list = [0.6 ** i for i in range(16)]
    net = M.DKiS(in_1=channels_in, in_2=channels_in, alpha_list=alpha_list)
    M.init_model(net)
    optim = torch.optim.Adam(net.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
    visualizer = viz.Visualizer(c.loss_names)
    train(net, optim, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer, expinfo=exp_info, lam=lam)





if __name__ == '__main__':

    attack_method = 'round'
    channels_in = 12
    lambda_c = 1
    lambda_s = 1
    lam = (lambda_c, lambda_s)
    exp_info = 'DKiS'

    print('----------start----------')

    k_queue = queue.Queue(maxsize=10)
    val_k_queue = queue.Queue(maxsize=4)
    t1 = threading.Thread(target=put_k, args=([16, c.batch_size // 2, 12, c.cropsize // 2, c.cropsize // 2], k_queue))
    t2 = threading.Thread(target=put_k,
                          args=([16, c.batchsize_val // 2, 12, c.cropsize_val // 2, c.cropsize_val // 2], val_k_queue))
    # t3 = threading.Thread(target=watch_k_queue, args=(k_queue, ))
    t1.setDaemon(True)
    t2.setDaemon(True)
    t1.start()
    t2.start()
    # t3.start()
    trainloader, testloader = datasets.get_dataset('DIV')

    iwt = utils.NIWT(a=0.43, b=0.28)    # a = traindataset.mean, b = traindataset.std
    dwt = utils.NDWT(a=0.43, b=0.28)    # div       a=0.43,  b=0.28
                                        # pub       a=0.904, b=0.215
                                        # ImageNet  a=0.434, b=0.275
                                        # COCO      a=0.44,  b=0.28

    main(attack_method, start_epoch=0, end_epoch=1600, lam=lam, exp_info=exp_info)
