import torch
import math

from torch import nn


class SelfAttUnit(nn.Module):
    def __init__(self, n_tokens, ratio=0.):
        super(SelfAttUnit, self).__init__()
        self.n_tokens = n_tokens
        self.n_head = 1
        self.d_k = int(self.n_tokens / self.n_head)
        self.d_v = int(self.n_tokens / self.n_head)
        self.w_q = nn.Linear(self.n_tokens, self.d_k, bias=False)
        self.w_k = nn.Linear(self.n_tokens, self.d_k, bias=False)
        self.w_v = nn.Linear(self.n_tokens, self.d_v, bias=False)
        self.fcn = nn.Sequential(
            nn.Linear(self.n_tokens, self.n_tokens, bias=False),
            nn.Dropout(ratio),
        )

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        att = torch.bmm(q, k.transpose(2, 1)).div(math.sqrt(self.d_k))
        att = torch.bmm(nn.functional.softmax(att, dim=1), v)
        return self.fcn(att)


class MLPUnit(nn.Module):
    def __init__(self, origin_dim, hidden_dim, ratio=0.):
        super(MLPUnit, self).__init__()
        self.entity = nn.Sequential(
            nn.Linear(origin_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(ratio),
            nn.Linear(hidden_dim, origin_dim),
            nn.Dropout(ratio),
        )

    def forward(self, x):
        return self.entity(x)


class MultiHeadSelfAttUnit(nn.Module):
    def __init__(self, n_tokens, n_head, ratio=0.):
        super(MultiHeadSelfAttUnit, self).__init__()
        self.n_tokens = n_tokens
        self.n_head = n_head
        self.d_k = int(self.n_tokens / self.n_head)
        self.d_v = int(self.n_tokens / self.n_head)
        self.weights = []
        for i in range(n_head):
            w_q = nn.Linear(self.n_tokens, self.d_k, bias=False)
            w_k = nn.Linear(self.n_tokens, self.d_k, bias=False)
            w_v = nn.Linear(self.n_tokens, self.d_v, bias=False)
            self.weights.append((w_q, w_k, w_v))
        self.w_o = nn.Linear(self.n_head * self.d_v, self.n_tokens, bias=False)
        self.fcn = nn.Sequential(
            nn.Linear(self.n_tokens, self.n_tokens, bias=False),
            nn.Dropout(ratio),
        )

    def forward(self, x):
        atts = []
        for i in range(len(self.weights)):
            w_q, w_k, w_v = self.weights[i]
            w_q.cuda()
            w_k.cuda()
            w_v.cuda()
            q = w_q(x)
            k = w_k(x)
            v = w_v(x)
            att = torch.bmm(q, k.transpose(2, 1).div(math.sqrt(self.d_k)))
            att = torch.bmm(nn.functional.softmax(att, dim=1), v)
            atts.append(att)
        ret = atts[0]
        for i in range(1, len(atts)):
            ret = torch.cat((ret, atts[i]), dim=2)
        return self.fcn(self.w_o(ret))


class TransformerBlock(nn.Module):
    def __init__(self, n_tokens, n_head, hidden_dim, ratio):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttUnit(n_tokens, n_head)
        self.fst_layer_norm = nn.LayerNorm(n_tokens)
        self.mlp = MLPUnit(n_tokens, hidden_dim, ratio)
        self.snd_layer_norm = nn.LayerNorm(n_tokens)

    def forward(self, x):
        identity = x
        x = self.fst_layer_norm(x)
        x = self.att(x) + identity
        identity = x
        x = self.snd_layer_norm(x)
        x = self.mlp(x) + identity
        return x
