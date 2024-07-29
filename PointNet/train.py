import torch

from tensorboardX import SummaryWriter


writer = SummaryWriter("./logs")
def training_loop(n_iterations, network, loader, loss_fn, optim):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 7000)
    network.train()

    losses = []
    for iteration in range(1, n_iterations + 1):
        pts, lbs = next(iter(loader))
        pts = pts.to("cuda", dtype=torch.float)
        lbs = lbs.to("cuda")

        output = network(pts)
        loss = loss_fn(output, lbs)

        optim.zero_grad()
        loss.backward()
        optim.step()

        temp = loss.item()
        losses.append(temp)
        writer.add_scalar("Losses", temp, global_step=(iteration))
        print("iter {}, loss {:.7f}".format(iteration, temp))

        scheduler.step()
        if iteration % 10000 == 0:
            network.eval().cpu()
            torch.save(network.state_dict(), "./ckpts/pointnet2_ssg_v{}.pth".format(iteration))
            network.train().cuda()

    return losses
