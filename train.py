import torch
import torch.optim as optim
from torch.utils.data import Dataloader, TensorDataset
import pandas as pd
from model import TradingVAE, loss
import argparse
import os

parser = argparse.ArgumentParser(description = "VAE training")
parser.add_argument('-e', help = "number of epochs", type = int, dest = 'e', default = 1)
parser.add_argumetn('-w', help = "path to load weights", type = str, dest = 'w')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if __name__ == "__main__":
    np_data = np.load("comp_dataset.npz")
    states = torch.from_numpy(np_data[:,:-3])
    dataset = TensorDataset(states)
    loader = Dataloader(dataset, batch_size = 32, pin_memory = True
                        shuffle = True, num_workers = 1)
 
    metadata = pd.read_csv("features.csv")
    meta = torch.Tensor(metadata[metadata.columns[1:].values).T
    meta = meta.to(device)
    vae = TradingVAE(meta)
    vae.to(device)
    if args.w:
        weights = torch.load(args.w, map_location = device)
        vae.load_state_dict(weights)
    vae.train()

    optimizer = optim.AdamW(vae.parameters(), lr = 1e-3)
    if os.path.exists("adam_w.pt"):
        optim_state = torch.load("adam_w.pt", map_location = device) 
        optimizer.load_state_dict(optim_state)        

    for epoch in range(args.e):
        running_loss = 0.
        for idx, inputs in enumerate(loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recons, mu, logvar = vae(inputs)
            loss = VAEloss(recons, inputs, mu, logvar)
            loss.backward(retain_graph = True)
            running_loss += loss
            optimizer.step()

            if idx % 1000 == 999:
                print(f"Epoch: {epoch} [{100.*idx/len(loader):0f}]\tLoss: {loss.item()/999:.5f}")

    torch.save(vae.model_state_dict(), f"vae_{args.e}.pt")
    torch.save(optimizer.model_state_dict(), "adam_w.pt")

