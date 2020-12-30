#!/usr/bin/env python3
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import TradingVAE, VAEloss
import argparse
import os

parser = argparse.ArgumentParser(description = "VAE training")
parser.add_argument('-e', help = "number of epochs", type = int, dest = 'e', default = 1)
parser.add_argument('-w', help = "path to load weights", type = str, dest = 'w')
parser.add_argument('-b', help = "batch size", type = int , dest = 'b', default = 64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if __name__ == "__main__":
    comp = np.load("comp_dataset.npz")
    states = torch.from_numpy(comp["state"])
    print(f"Data length: {states.shape[0]}")
    loader = DataLoader(states, batch_size = args.b, pin_memory = True,
                        shuffle = True, num_workers = 1)
 
    metadata = pd.read_csv("features.csv")
    meta = torch.Tensor(metadata[metadata.columns[1:]].values).T
    meta = meta.to(device)
    vae = TradingVAE(meta)
    vae.to(device)

    if args.w:
        weights = torch.load(args.w, map_location = device)
        vae.load_state_dict(weights)
    vae.train()

    optimizer = optim.AdamW(vae.parameters(), 
                            lr = 1e-3, 
                            amsgrad = True)

    if os.path.exists(os.path.join("weights", "adam_w.pt")):
        optim_state = torch.load("adam_w.pt", map_location = device) 
        optimizer.load_state_dict(optim_state)        

    for epoch in range(args.e):
        stats = []
        running_loss = 0.
        rec_loss = 0.
        for idx, inputs in enumerate(loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recons, mu, logvar = vae(inputs)
            loss, recon_loss = VAEloss(recons, inputs, mu, logvar)
            loss.backward(retain_graph = True)
            optimizer.step()

            stats.append(loss)
            running_loss += loss
            rec_loss += recon_loss

            if idx % 1000 == 999: 
                avg_loss = running_loss.item()/1000
                print(f"Epoch: {epoch} [{100.*idx/len(loader):.0f}%]\tLoss: {avg_loss:.3f}"
                      f"\tReconstruction Loss: {recon_loss.item()/1000:.6f}")
                running_loss = 0.
                rec_loss = 0.

        with open("stats.txt", 'a+') as f:
            f.write(f"Epoch {epoch}\n")
            f.write( ','.join( [ format(l, '.3f') for l in stats]))
            f.write('\n')

    torch.save(vae.state_dict(), f"vae_{args.e}.pt")
    torch.save(optimizer.state_dict(), "adam_w.pt")

