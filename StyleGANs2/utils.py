import torch

from torch.autograd import Variable, grad


def calc_gradient_penalty(x, y, f):
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device=x.device)
    z = x + alpha * (y - x)
    z = Variable(z, requires_grad=True).to(x.device)
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.shape).to(z.device), create_graph=True)[0].view(z.shape[0], -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
    return gp
