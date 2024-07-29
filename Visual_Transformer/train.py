import torch

from tensorboardX import SummaryWriter

writer = SummaryWriter("./logs")
def training_loop(n_epochs, net, loss_fn, loader, optim):
    total_iter = 0

    net.train().cuda()
    for epoch in range(n_epochs + 1):
        for i, (imgs, lbes) in enumerate(loader, start=0):
            b, _, _, _ = imgs.shape
            imgs = imgs.to("cuda", dtype=torch.float)
            lbes = lbes.to("cuda")

            output = net(imgs)
            loss = loss_fn(output, lbes)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_iter = total_iter + 1
            writer.add_scalar("loss", loss.item(), global_step=total_iter)

            print("eopch {}\t iter {}\t loss {:.6f}\t".format(epoch, i, loss.item()))

        net.eval().cpu()
        torch.save(net.state_dict(), "./ckpts/classifier_{}.pth".format(epoch + 1))
        net.train().cuda()
