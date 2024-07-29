import torch


def training_loop(n_epochs, loader, sphere, generator, discriminator, loss_fn, gene_optim, disc_optim):
    for epoch in range(1, n_epochs):
        for i, (pc, _) in enumerate(loader, start=0):
            b, _, _ = pc.shape
            pc = pc.to("cuda", dtype=torch.float)

            # * handle discriminator
            disc_optim.zero_grad()
            r_shape_scr = discriminator(pc)
            r_shape_lbe = torch.full((b, 1), 1., dtype=torch.float, device="cuda")
            r_disc_loss = loss_fn(r_shape_scr, r_shape_lbe) / 2
            r_disc_loss.backward()

            global_prior = sphere
            global_prior = global_prior.expand((b, -1, -1)).to("cuda", dtype=torch.float)
            fpc = generator(global_prior)
            f_shape_scr = discriminator(fpc.detach())
            f_shape_lbe = torch.full((b, 1), 0., dtype=torch.float, device="cuda")
            f_disc_loss = loss_fn(f_shape_scr, f_shape_lbe) / 2
            f_disc_loss.backward()
            disc_loss = r_disc_loss + f_disc_loss
            disc_optim.step()

            # * handle generator
            gene_optim.zero_grad()
            f_shape_scr = discriminator(fpc)
            c_gene_loss = loss_fn(f_shape_scr, r_shape_lbe)
            c_gene_loss.backward()
            gene_loss = c_gene_loss
            gene_optim.step()

            print("epoch {}\t iter {}\t disc {:.6f}\t gene {:.4f}".format(epoch, i, disc_loss.item(), gene_loss.item()))

        generator.eval().cpu()
        torch.save(generator.state_dict(), "./ckpts/chair/generator_v{}.pth".format(epoch))
        generator.train().cuda()
