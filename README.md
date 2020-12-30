# Jane Street Market Prediction

## Data
npz data:
* data["states"] has columns (features 0 -- 129, weight, ts_id, date)
* data["resps"] has columns (resp, resp 1 -- 4)
processing:
* fill incomplete data with gaussian or regression

## Encoder
* Variational autoencoder
* use metadata features in 1d conv

## RNN Model
* Try vanilla GRU/LSTM:
    + input sequence --> VAE --> RNN --> next action 
* sequence to sequence model:
    + input sequence --> VAE --> RNN --> action sequence 
* Transformer (Attention is All You Need)
