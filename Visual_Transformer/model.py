import torch

from torch import nn

from components import TransformerBlock


class VisualTransformer(nn.Module):
    def __init__(self, n_block, n_tokens, patch_size, n_sequence, n_head, hidden_dim, k, ratio=0.):
        super(VisualTransformer, self).__init__()
        # * hyper-parameters
        self.patch_size = patch_size
        self.n_tokens = n_tokens
        # * projection, token & embedding
        self.projection = nn.Conv2d(3, n_tokens, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_tokens))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_sequence + 1, n_tokens))
        # * transformer blocks, last mlp
        self.transformer_encoder = nn.Sequential()
        for i in range(n_block):
            self.transformer_encoder.add_module("transformer_{}".format(i), 
                TransformerBlock(n_tokens, n_head, hidden_dim, ratio))
        self.mlp_head = nn.Linear(n_tokens, k)

    def forward(self, x):
        b, _, _, _ = x.shape
        patches = self.projection(x).reshape(b, self.n_tokens, -1).transpose(2, 1)
        patches = torch.cat((patches, self.cls_token.expand(b, 1, -1)), dim=1)
        patches = patches + self.pos_embedding.expand(b, -1, -1)
        output = self.mlp_head(self.transformer_encoder(patches)[:, 0, :])
        return output
