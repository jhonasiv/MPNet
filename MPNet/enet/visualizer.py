import argparse
import os

import numpy as np
import torch
from plotly import graph_objs as go
from plotly.subplots import make_subplots

import data_loader as dl
from MPNet.enet.CAE import ContractiveAutoEncoder

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def load_env(qtt):
    return dl.load_perms(qtt, np.random.randint(55000, 70000, 1)[0])


def plot(qtt, samples, enet, title):
    num_rows = int(np.ceil(qtt / 3))
    num_cols = 3 if num_rows != 1 else qtt
    fig = make_subplots(rows=num_rows, cols=num_cols, shared_xaxes=True, shared_yaxes=True)
    results = enet.reconstruction(torch.from_numpy(samples).float())
    for n, (sample, result) in enumerate(zip(samples, results)):
        sample = sample.reshape((-1, 2))
        x = sample[:, 0]
        y = sample[:, 1]
        fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers', marker=dict(color='black'), name="Obstáculos",
                           legendgroup="obst", showlegend=n == 0), row=(int(n / 3) + 1), col=(n % 3) + 1)
        
        result = result.numpy().reshape((-1, 2))
        fig.add_trace(go.Scatter(x=result[:, 0], y=result[:, 1], mode='markers', marker=dict(color="green"),
                                 name="Representação<br>reconstruída", legendgroup="enet", showlegend=n == 0),
                      row=(int(n / 3) + 1), col=(n % 3) + 1)
    fig.update_xaxes(range=[-20, 20], tickmode='array', tickvals=[-20, 20], ticktext=[-20, 20], showgrid=False,
                     zeroline=False)
    fig.update_yaxes(range=[-20, 20], tickmode='array', tickvals=[-20, 20], ticktext=[-20, 20], showgrid=False,
                     zeroline=False)
    fig.update_layout(title=title, legend=dict(itemsizing='constant'))
    fig.show()


def main(args):
    envs = load_env(args.num)
    samples = np.array(list(map(dl.create_samples, envs)))
    for enet_ckpt in args.enets:
        enet = ContractiveAutoEncoder.load_from_checkpoint(enet_ckpt)
        enet.freeze()
        plot(args.num, samples, enet, "Representações reconstruídas do módulo ENet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=5, type=int)
    parser.add_argument('--enets', nargs='+', default=[], required=True)
    
    main(parser.parse_args())
