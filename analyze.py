import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

from model import *

def calc_k(model, x):
    n_batch = x.shape[0]
    x_0 = model.embed[x,:] + model.pos_embed
    k = torch.einsum('hdm,btm->bhdt', model.W_k, x_0)
    return k

def calc_attn(model, x):
    n_batch = x.shape[0]
    x_0 = model.embed[x,:] + model.pos_embed
    k = torch.einsum('hdm,btm->bhdt', model.W_k, x_0)
    q = torch.einsum('hdm,bm->bhd', model.W_q, x_0[:,-1,:])
    A = torch.softmax(torch.einsum('bhdt,bhd->bht', k, q), dim=2)
    return A

def final_embed():
    with torch.no_grad():
        epoch = n_epochs - 1
        state_dict = torch.load(f'models/model_{epoch:05d}.pt', map_location=device)
        model = SingleLayerTransformer().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        embed = model.embed.cpu().numpy()[:P,:]
        pca = PCA(n_components=10)
        components = pca.fit_transform(embed)
        for i in range(10):
            plt.scatter(np.arange(P), components[:,i])
        plt.show()

def final_embed_fft():
    with torch.no_grad():
        epoch = n_epochs - 1
        state_dict = torch.load(f'models/model_{epoch:05d}.pt', map_location=device)
        model = SingleLayerTransformer().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        embed = model.embed.cpu().numpy()[:P,:]
        fft = np.fft.fft(embed, axis=0)
        fft_r = 0.5 + np.real(fft)
        fft_g = 0.5 + np.imag(fft)
        fft_rgb = np.stack((fft_r, fft_g, np.zeros_like(fft_r) + 0.5), axis=2)
        plt.imshow(np.abs(fft_rgb))
        plt.show()

def final_attn():
    with torch.no_grad():
        epoch = n_epochs - 1
        state_dict = torch.load(f'models/model_{epoch:05d}.pt', map_location=device)
        model = SingleLayerTransformer().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print(model)

        a_values = torch.arange(0, P)
        b_values = torch.arange(0, P)
        a_square, b_square = torch.meshgrid(a_values, b_values, indexing='ij')
        eq_square = torch.full((P, P), P)
        x_data = torch.stack((a_square, b_square, eq_square), dim=2).reshape(P * P, 3)
        x = x_data.to(device)

        A = calc_attn(model, x).reshape(P, P, 4, 3)
        fig, ax = plt.subplots(4, 3)
        for i in range(4):
            for j in range(3):
                ax[i,j].imshow(A[:,:,i,j].cpu())
        plt.show()

        # k = calc_k(model, x)
        # for component in range(32):
        #     im = k[:,0,component,0].reshape(P,P)[:,0].cpu().numpy()
        #     fft = np.fft.fft(im)

        #     plt.bar(np.arange(1,P), np.abs(fft[1:]))
        # plt.show()

def final_attn_fft():
    with torch.no_grad():
        epoch = n_epochs - 1
        state_dict = torch.load(f'models/model_{epoch:05d}.pt', map_location=device)
        model = SingleLayerTransformer().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print(model)

        a_values = torch.arange(0, P)
        b_values = torch.arange(0, P)
        a_square, b_square = torch.meshgrid(a_values, b_values, indexing='ij')
        eq_square = torch.full((P, P), P)
        x_data = torch.stack((a_square, b_square, eq_square), dim=2).reshape(P * P, 3)
        x = x_data.to(device)

        A = calc_attn(model, x).reshape(P, P, 4, 3)
        fig, ax = plt.subplots(4, 3)
        for i in range(4):
            for j in range(3):
                fft = np.fft.fft2(A[:,:,i,j].cpu().numpy())
                ax[i,j].imshow(np.abs(fft), vmax=1)
        plt.show()
