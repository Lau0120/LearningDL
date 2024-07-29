import time
import os
import torch
import torch.nn.functional as F

from config import hp
from utils import calc_gram_matrix, normalize_batch


def training_loop(n_epochs, loader, transfer_net, optimizer, loss_net, cw, sw, gram_styl):
    for e in range(n_epochs):
        transfer_net.train()

        cont_losses = 0.
        styl_losses = 0.
        count = 0
        for batch_id, (x, _) in enumerate(loader, start=0):
            n_batch = len(x)
            count = count + n_batch
            optimizer.zero_grad()

            # ! forward pass
            x = x.to(hp.device)
            y = transfer_net(x)

            x = normalize_batch(x)
            y = normalize_batch(y)

            features_x = loss_net(x)
            features_y = loss_net(y)

            # ! calculate cont & styl loss
            cont_loss = cw * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            styl_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_styl):
                gm_y = calc_gram_matrix(ft_y)
                styl_loss = styl_loss + F.mse_loss(gm_y, gm_s)
            styl_loss = sw * styl_loss

            tota_loss = cont_loss + styl_loss
            tota_loss.backward()
            optimizer.step()

            # ! print optimization log
            cont_losses = cont_losses + cont_loss.item()
            styl_losses = styl_losses + styl_loss.item()
            if (batch_id + 1) % hp.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, hp.sample_size,
                                  cont_losses / (batch_id + 1),
                                  styl_losses / (batch_id + 1),
                                  (cont_losses + styl_losses) / (batch_id + 1)
                )
                print(mesg)

            # ! save transfer net parameters
            if (batch_id + 1) % hp.ckpt_interval == 0:
                transfer_net.eval().cpu()
                torch.save(transfer_net.state_dict(), os.path.join(hp.ckpt_dirs, "transfer_oil_net_e{}b{}.pth".format(e + 1, batch_id + 1)))
                transfer_net.to(hp.device).train()
