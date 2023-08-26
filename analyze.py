import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

from model import *

def final_embed():
    with torch.no_grad():
        epoch = n_epochs - 1
        state_dict = torch.load(f'models/model_{epoch:05d}.pt', map_location=device)
        model = SingleLayerTransformer().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print(model)

        embeddings = model.embed[:P,:].cpu().numpy()
        print(embeddings.shape)
        fft = np.fft.fft(embeddings, axis=0)
        print(fft.shape)

        plt.imshow(np.abs(embeddings), cmap='gray')
        plt.show()