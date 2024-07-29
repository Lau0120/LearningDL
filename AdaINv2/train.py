import torch
import time
import torch.nn.functional as F

from utils import AdaIN


def training_loop(n_epochs, cont_loader, styl_loader, encoder, decoder, optimizer, styl_factor):
    for e in range(n_epochs):
        decoder.train()

        cont_losses = 0.
        styl_losses = 0.
        count = 0
        for batch_id, ((conts_x, _), (styls_x, _)) in enumerate(zip(cont_loader, styl_loader), start=0):
            count = count + len(conts_x)

            # ! forward
            conts_x = conts_x.float().to(torch.device("cuda"))
            styls_x = styls_x.float().to(torch.device("cuda"))

            optimizer.zero_grad()

            conts_feature_maps = encoder(conts_x)
            styls_feature_maps = encoder(styls_x)
            syncs_t = AdaIN(conts_feature_maps[-1], styls_feature_maps[-1])
            outputs = decoder(syncs_t)

            # ! calculate loss
            outpt_feature_maps = encoder(outputs)

            cont_loss = F.mse_loss(outpt_feature_maps[-1], syncs_t)
            styl_loss = 0.
            for outpt_relu, styls_relu in zip(outpt_feature_maps, styls_feature_maps):
                temp_loss = F.mse_loss(outpt_relu.mean((2, 3), keepdim=True),
                                       styls_relu.mean((2, 3), keepdim=True)) + \
                            F.mse_loss(outpt_relu.std((2, 3), keepdim=True),
                                       styls_relu.std((2, 3), keepdim=True))
                styl_loss = styl_loss + temp_loss

            # ! backward
            tota_loss = cont_loss + styl_loss * styl_factor
            tota_loss.backward()
            optimizer.step()

            # ! print optimization log
            cont_losses = cont_losses + cont_loss.item()
            styl_losses = styl_losses + styl_loss.item()
            if (batch_id + 1) % 1 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, 70000,
                    cont_losses / (batch_id + 1),
                    styl_losses / (batch_id + 1),
                    (cont_losses + styl_losses) / (batch_id + 1),
                )
                print(mesg)

            # ! save transfer net parameters
            if (batch_id + 1) % 1000 == 0:
                decoder.eval().cpu()
                torch.save(decoder.state_dict(), "./ckpt_nets/adain_transfer_e{}b{}.pth".format(e + 1, batch_id + 1))
                decoder.train().to(torch.device("cuda"))
