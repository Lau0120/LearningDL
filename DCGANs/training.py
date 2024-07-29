import torch

from config import device
from torchvision.utils import make_grid


def training_loop(epochs, loader, generator, discriminator, loss_fn, optimizerD, optimizerG, imagesL, lossesG, lossesD):
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    iters = 0
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(loader, 0):
            # >>> update D network: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()
            real_imgs = imgs.to(device)
            b_size = imgs.size(0)
            labels = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
            output = discriminator(real_imgs).view(-1)
            lossD_in_real = loss_fn(output, labels)
            lossD_in_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake_imgs = generator(noise)
            labels.fill_(fake_label)
            output = discriminator(fake_imgs.detach()).view(-1)
            lossD_in_fake = loss_fn(output, labels)
            lossD_in_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_in_real + lossD_in_fake
            optimizerD.step()


            # >>> update G network: maximize log(D(G(z)))
            generator.zero_grad()
            labels.fill_(real_label)
            output = discriminator(fake_imgs).view(-1)
            lossG = loss_fn(output, labels)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()


            # >>> output training log
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(loader),
                        lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

                lossesG.append(lossG.item())
                lossesD.append(lossD.item())

                if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(loader) - 1)):
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                    imagesL.append(make_grid(fake, padding=2, normalize=True))

            iters += 1
