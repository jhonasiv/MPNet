from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os
from CAE import ContractiveAutoEncoder
from plotly import graph_objs as go
import pytorch_lightning as pl
import data_loader as dl
from torch import nn

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def plot(inp, output, title):
    inp = inp.cpu().detach().numpy().reshape((-1, 2))
    output = output.cpu().detach().numpy().reshape((-1, 2))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=inp[:, 0], y=inp[:, 1], mode='markers', marker={'color': 'black', 'size': 5}))
    fig.add_trace(go.Scatter(x=output[:, 0], y=output[:, 1], mode='markers', marker={'color': 'blue', 'size': 5}))
    fig.update_xaxes(range=[-20, 20])
    fig.update_yaxes(range=[-20, 20])
    fig.update_layout(title=title)
    fig.show()


def train(args):
    if not os.path.exists(f"{project_path}/{args.model_path}"):
        os.makedirs(f"{project_path}/{args.model_path}")
    
    pl.seed_everything(42)
    training = dl.loader(55000, args.batch_size, 0)
    validation = dl.loader(8250, 1, 55000)
    test = dl.loader(5000, 1, 63250)
    
    cae = ContractiveAutoEncoder(training, validation, config={'actv': nn.SELU}, test_dataloader=test)
    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='min', verbose=True)
    checkpointing = ModelCheckpoint(monitor='val_loss', dirpath=f"{project_path}/models/", filename="cae",
                                    verbose=True, save_top_k=3)
    
    trainer = pl.Trainer(gpus=1, auto_select_gpus=True, callbacks=[es, checkpointing],
                         stochastic_weight_avg=True, deterministic=True, benchmark=True)
    
    trainer.fit(cae)
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=60)
    parser.add_argument('--model_path', default="models")
    parser.add_argument('--batch_size', default=100)
    
    args = parser.parse_args()
    
    train(args)
