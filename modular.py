import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import torch

from model import *
import analyze

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

os.makedirs('models', exist_ok=True)

if sys.argv[1:] == ['final','embed']:
    analyze.final_embed()
elif sys.argv[1:] == ['final','embed_fft']:
    analyze.final_embed_fft()
elif sys.argv[1:] == ['final','attn']:
    analyze.final_attn()
elif sys.argv[1:] == ['final','attn_fft']:
    analyze.final_attn_fft()
elif sys.argv[1:] == ['final', 'logits']:
    analyze.final_logits()
elif sys.argv[1:] == ['final', 'probs']:
    analyze.final_probs()
elif sys.argv[1:] == ['train']:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    train_losses = np.zeros(n_epochs // save_every)
    test_losses = np.zeros(n_epochs // save_every)
    sum_train_losses = 0
    sum_test_losses = 0

    start_time = time.monotonic()

    try:
        for epoch in range(n_epochs):
            model.train()
            train_data_shuffled = train_data[torch.randperm(n_train)]
            batch_size = 256
            loss_sum = torch.tensor(0.0).to(device)
            for i in range(0, n_train, batch_size):
                optimizer.zero_grad()
                batch = train_data_shuffled[i:i+batch_size]
                logits = model(batch[:,:3])
                loss = loss_fn(logits, batch[:,-1])
                loss_sum += loss.detach().sum()
                loss.backward()
                optimizer.step()
            sum_train_losses += loss_sum.item() / n_train

            model.eval()
            with torch.no_grad():
                test_logits = model(test_data[:,:3])
                test_loss = loss_fn(test_logits, test_data[:,-1]).item()
                sum_test_losses += test_loss / n_test

            if epoch % save_every == save_every - 1:
                avg_train_loss = sum_train_losses / save_every
                avg_test_loss = sum_test_losses / save_every
                train_losses[epoch // save_every] = avg_train_loss
                test_losses[epoch // save_every] = avg_test_loss
                filename = f'models/model_{epoch:05d}.pt'
                torch.save(model.state_dict(), filename)
                timing = time.monotonic() - start_time
                print(f'{timing:8.2f}: epoch: {epoch:5d}   train loss: {avg_train_loss:12.4f}   test loss: {avg_test_loss:12.4f}')
                sum_train_losses = 0
                sum_test_losses = 0

    except KeyboardInterrupt:
        pass

    plt.semilogy(train_losses[:epoch//save_every], label='train loss')
    plt.semilogy(test_losses[:epoch//save_every], label='test loss')
    plt.legend()
    plt.show()
else:
    print("Please supply a valid subcommand.")