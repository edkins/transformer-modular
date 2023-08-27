import torch
from torch.nn.functional import softmax, relu

seed = 0
P = 113
n_train = (P * P * 3) // 10
n_epochs = 25000
save_every = 50
weight_decay = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SingleLayerTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_vocab = P + 1 # v
        self.d_model = 128   # m
        self.d_head = 32     # d
        self.d_mlp = 512     # p
        self.n_head = 4      # h
        self.window_size = 3 # t

        scale = 30

        self.embed = torch.nn.Parameter(torch.randn((self.n_vocab, self.d_model)) / scale)
        self.pos_embed = torch.nn.Parameter(torch.randn((1, self.window_size, self.d_model)) / scale)
        self.W_k = torch.nn.Parameter(torch.randn((self.n_head, self.d_head, self.d_model)) / scale)
        self.W_q = torch.nn.Parameter(torch.randn((self.n_head, self.d_head, self.d_model)) / scale)
        self.W_v = torch.nn.Parameter(torch.randn((self.n_head, self.d_head, self.d_model)) / scale)
        self.W_o = torch.nn.Parameter(torch.randn((self.n_head, self.d_model, self.d_head)) / scale)
        self.W_in = torch.nn.Parameter(torch.randn((self.d_mlp, self.d_model)) / scale)
        self.W_out = torch.nn.Parameter(torch.randn((self.d_model, self.d_mlp)) / scale)
        self.unembed = torch.nn.Parameter(torch.randn((self.n_vocab, self.d_model)) / scale)

    def forward(self, x):
        n_batch = x.shape[0]
        x_0 = self.embed[x,:] + self.pos_embed
        k = torch.einsum('hdm,btm->bhdt', self.W_k, x_0)
        q = torch.einsum('hdm,bm->bhd', self.W_q, x_0[:,-1,:])
        A = torch.softmax(torch.einsum('bhdt,bhd->bht', k, q), dim=2)
        Ax0 = torch.einsum('bht,btm->bhm', A, x_0)
        v = torch.einsum('hdm,bhm->bhd', self.W_v, Ax0)
        x_1 = torch.einsum('hmd,bhd->bm', self.W_o, v) + x_0[:,-1,:]
        MLP = relu(torch.einsum('pm,bm->bp', self.W_in, x_1))
        x_2 = torch.einsum('mp,bp->bm', self.W_out, MLP) + x_1
        logits = torch.einsum('vm,bm->bv', self.unembed, x_2)
        return logits
