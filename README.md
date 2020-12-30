# Jane Street Market Prediction

## Data
[npz data](https://drive.google.com/file/d/17zcBLQxuWv_vQFJj8ZmRLJcO6w2xNe65/view?usp=sharing) (1 GB)

`data = np.load("js_data.npz")`
* data["states"] has columns (features 0 -- 129, weight)
* data["date"] has columns (ts_id, date)
* data["resps"] has columns (resp, resp 1 -- 4)
* data["meta"] boolean metadata

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
