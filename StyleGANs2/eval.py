import torch

from torchvision.utils import save_image, make_grid
from model import StyleGenerator


z = torch.randn((5, 512), device="cuda")
outputs = []
for vi in range(80):
    generator = StyleGenerator()
    state = torch.load("./ckpts/style_gan_generator_v{}.pth".format(vi + 1))
    generator.load_state_dict(state)
    generator.eval().cuda()
    _, output = generator(z)
    output = output.chunk(5, dim=0)
    for out in output:
        outputs.append(out.squeeze(0))
grid = make_grid(outputs, nrow=5)
save_image(grid, "./images/result.jpg")
