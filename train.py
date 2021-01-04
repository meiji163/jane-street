#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import TradingRNN, TradingData, greedy_seq
import argparse
import sys
import os

parser = argparse.ArgumentParser(description = "RNN training")
parser.add_argument('-e', help = "number of epochs", type = int, dest = 'e', default = 1)
parser.add_argument('-w', help = "path to weights", type = str, dest = 'w')
parser.add_argument('-b', help = "batch size", type = int , dest = 'b', default = 64)
parser.add_argument('-l', help = "sequence length", type = int, dest = 'l', default = 200)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if __name__ == "__main__":
    print("Loading data...", end = '')
    sys.stdout.flush() 
    data = TradingData(os.path.join("data","dataset.npz"), 
                       seq_len = args.l)
    print("done")
    loader = DataLoader(data, batch_size = args.b, pin_memory = True,
                        shuffle = True, num_workers = 2)
    rnn = TradingRNN(data.meta.to(device), n_lyrs = 1)

    if args.w:
        weights = torch.load(args.w, map_location = device)
        rnn.load_state_dict(weights)
    rnn.to(device)
    rnn.train()

    xent = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(rnn.parameters(), 
                            lr = 5e-3,
                            amsgrad = True)
    if os.path.exists(os.path.join("weights", "adam_w.pt")):
        optim_state = torch.load(os.path.join("weights", "adam_w.pt"),
                                 map_location = device) 
        optimizer.load_state_dict(optim_state)        

    for epoch in range(args.e):
        stats = []
        running_loss = 0.
        for idx, inputs in enumerate(loader):
            fts, wts, targs = inputs
            fts, wts, targs = fts.to(device), wts.to(device), targs.to(device)
            out, _ = rnn(fts, wts)
            loss = xent(torch.transpose(out,1,2), targs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            stats.append(loss)
            running_loss += loss
            if idx % 1000 == 999: 
                avg_loss = running_loss.item()/1000
                print(f"Epoch: {epoch} [{100.*idx/len(loader):.0f}%]\tLoss: {avg_loss:.4f}")
                running_loss = 0.

        with open("stats.txt", 'a+') as f:
            f.write(f"Epoch {epoch}\n")
            f.write( ','.join( [ format(l, '.3f') for l in stats]))
            f.write('\n')

    torch.save(rnn.state_dict(), f"rnn_{args.e}.pt")
    torch.save(optimizer.state_dict(), "adam_w.pt")

