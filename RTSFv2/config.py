import torch

from collections import namedtuple


HyperParameters = namedtuple("HyperParameters",
    [
        "image_size",
        "batch_size",
        "device",
        "n_epochs",
        "learning_rate",
        "cont_weight",
        "styl_weight",
        "dataset",
        "style_image",
        "log_interval",
        "sample_size",
        "ckpt_interval",
        "ckpt_dirs",
    ],
)

hp = HyperParameters(
    256,
    7,
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    2,
    1e-3,
    1e5,
    1e10,
    "G:/Project/Deep_Learning/Datasets/COCOs/train2014",
    "G:/Project/Deep_Learning/Style_Transfer/Images/Style/oil.jpg",
    1,
    70000,
    100,
    "G:/Project/Deep_Learning/Style_Transfer/RTSFv2/ckpt_nets",
)
