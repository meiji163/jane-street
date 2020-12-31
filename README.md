# Jane Street Market Prediction

## Data
[npz data](https://drive.google.com/file/d/1BVKSILxLOk45f8Zxh3cS2l_E-vNaTA2y/view?usp=sharing)

`data = np.load("js_data.npz")`
* "states" has columns (features 0 -- 129, weight)
* "date" has columns (ts_id, date)
* "resps" has columns (resp, resp 1 -- 4)
* "meta" boolean metadata
* "mean"/"std" has mean/standard dev for features 1 -- 129

processing:
* normalize?
* fill incomplete data with gaussian or regression

## Encoder
* Variational autoencoder
* use metadata features in 1d conv
* can generate more data/interpolate

## RNN Model
* Try vanilla GRU/LSTM:
    + input sequence --> VAE --> RNN --> next action 
* sequence to sequence model:
    + input sequence --> VAE --> RNN --> action sequence 
    + add attention mechanism
* Transformer (Attention is All You Need)
