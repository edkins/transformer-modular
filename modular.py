import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.nn.functional import softmax, relu

seed = 0
P = 113
n_train = (P * P * 3) // 10
n_epochs = 20_000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

class SingleLayerTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_vocab = P + 1 # v
        self.d_model = 128   # m
        self.d_head = 32     # d
        self.d_mlp = 512     # p
        self.n_head = 4      # h
        self.window_size = 3 # t

        self.embed = torch.nn.Parameter(torch.randn((self.n_vocab, self.d_model)))
        self.pos_embed = torch.nn.Parameter(torch.randn((1, self.window_size, self.d_model)))
        self.W_k = torch.nn.Parameter(torch.randn((self.n_head, self.d_head, self.d_model)))
        self.W_q = torch.nn.Parameter(torch.randn((self.n_head, self.d_head, self.d_model)))
        self.W_v = torch.nn.Parameter(torch.randn((self.n_head, self.d_head, self.d_model)))
        self.W_o = torch.nn.Parameter(torch.randn((self.n_head, self.d_model, self.d_head)))
        self.W_in = torch.nn.Parameter(torch.randn((self.d_mlp, self.d_model)))
        self.W_out = torch.nn.Parameter(torch.randn((self.d_model, self.d_mlp)))
        self.unembed = torch.nn.Parameter(torch.randn((self.n_vocab, self.d_model)))

    def forward(self, x):
        n_batch = x.shape[0]
        x_0 = self.embed[x,:] + self.pos_embed
        k = torch.einsum('hdm,btm->bhdt', self.W_k, x_0)
        q = torch.einsum('hdm,bm->bhd', self.W_q, x_0[:,-1,:])
        A = torch.softmax(torch.einsum('bhdt,bhd->bht', k, q), dim=2)
        v = torch.einsum('hdm,btm->bhdt', self.W_v, x_0)
        x_1 = torch.einsum('hmd,bhdt->bm', self.W_o, v) + x_0[:,-1,:]
        MLP = relu(torch.einsum('pm,bm->bp', self.W_in, x_1))
        x_2 = torch.einsum('mp,bp->bm', self.W_out, MLP)
        logits = torch.einsum('vm,bm->bv', self.unembed, x_2)
        return logits

a_values = torch.arange(0, P)
b_values = torch.arange(0, P)
a_square, b_square = torch.meshgrid(a_values, b_values, indexing='ij')
eq_square = torch.full((P, P), P)
c_square = (a_square + b_square) % P
all_data = torch.stack((a_square, b_square, eq_square, c_square), dim=2).reshape(P * P, 4)
all_data_shuffled = all_data[torch.randperm(P * P)]

train_data = all_data_shuffled[:n_train].to(device)
test_data = all_data_shuffled[n_train:].to(device)
n_test = P * P - n_train

model = SingleLayerTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=10)
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

try:
    for epoch in range(n_epochs):
        model.train()
        train_data_shuffled = train_data[torch.randperm(n_train)]
        batch_size = 64
        loss_sum = torch.tensor(0.0).to(device)
        for i in range(0, n_train, batch_size):
            optimizer.zero_grad()
            batch = train_data_shuffled[i:i+batch_size]
            logits = model(batch[:,:3])
            loss = loss_fn(logits, batch[:,-1])
            loss_sum += loss.sum()
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            test_logits = model(test_data[:,:3])
            test_loss = loss_fn(test_logits, test_data[:,-1])
            train_losses[epoch] = loss_sum.item() / n_train
            test_losses[epoch] = test_loss.item() / n_test
            print(f'epoch: {epoch:5d}   train loss: {loss_sum.item() / n_train:12.4f}   test loss: {test_loss.item() / n_test:12.4f}')
except KeyboardInterrupt:
    pass

plt.semilogy(train_losses[:epoch], label='train loss')
plt.semilogy(test_losses[:epoch], label='test loss')
plt.legend()
plt.show()
