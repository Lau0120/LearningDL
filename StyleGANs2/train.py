import torch

from torch.optim import lr_scheduler

from utils import calc_gradient_penalty


def training_loop(n_epochs, loader, gene, gene_optim, disc, disc_optim):
    gene_scheduler = lr_scheduler.ExponentialLR(gene_optim, gamma=0.99)
    disc_scheduler = lr_scheduler.ExponentialLR(disc_optim, gamma=0.99)

    for epoch in range(1, n_epochs):
        for i, (imgs, _) in enumerate(loader, start=1):
            b, _, _, _ = imgs.shape
            real_imgs = imgs.to("cuda", dtype=torch.float)

            # handle discriminator
            disc_optim.zero_grad()
            # calculate real part
            disc_real_logit = disc(real_imgs)
            disc_real_loss = disc_real_logit.mean()
            # calculate fake part
            _, fake_imgs = gene(torch.randn((b, 512), device="cuda"))
            disc_fake_logit = disc(fake_imgs.detach())
            disc_fake_loss = disc_fake_logit.mean()
            # total loss
            disc_loss = -(disc_real_loss - disc_fake_loss) + calc_gradient_penalty(real_imgs.data, fake_imgs.data, disc) * 10.0
            # backward & next step
            disc_loss.backward()
            disc_optim.step()

            # handle generator
            if i % 5 == 0:
                gene_optim.zero_grad()
                # mixing part
                gene_mix_logit = disc(fake_imgs)
                gene_loss = -gene_mix_logit.mean()
                # backward & next step
                gene_loss.backward()
                gene_optim.step()

                # print logs
                print("epoch {}\titer {}\tdisc {:.4f}\tgene {:.4f}".format(epoch, i, disc_loss.item(), gene_loss.item()))

        # update scheduler
        disc_scheduler.step()
        gene_scheduler.step()

        # save generator weights
        torch.save(gene.state_dict(), "./ckpts/style_gan_generator_v{}.pth".format(epoch))
