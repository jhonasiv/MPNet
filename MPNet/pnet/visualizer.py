import argparse
import os

import numpy as np
import torch
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.enet.data_loader import create_samples
from MPNet.pnet.model import PNet

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def plot_path(perm, mpnet_path, reference_path):
    fig = go.Figure()
    draw_obstacles(fig, perm)
    
    x = mpnet_path[:, 0]
    y = mpnet_path[:, 1]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='mpnet'))
    x = reference_path[:, 0]
    y = reference_path[:, 1]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='rrts'))
    fig.update_xaxes(range=[-20, 20])
    fig.update_yaxes(range=[-20, 20])

    fig.show()


def draw_obstacles(fig, perm, **kwargs):
    obstacles = []
    for obstacle in perm:
        x, y = obstacle
        obstacles.extend([[x - 2.5, y - 2.5], [x - 2.5, y + 2.5], [x + 2.5, y + 2.5], [x + 2.5, y - 2.5],
                          [x - 2.5, y - 2.5], [None, None]])
    obstacles = np.array(obstacles)
    x = obstacles[:, 0]
    y = obstacles[:, 1]
    fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor="black", name="obstacle"), **kwargs)


def plot_sample(center_list, path_list):
    fig = make_subplots(rows=2, cols=3)
    for n, (center, paths) in enumerate(zip(center_list, path_list)):
        draw_obstacles(fig, center, row=(int(n / 3) + 1), col=(n % 3) + 1)
        
        for path in paths:
            x = path[:, 0]
            y = path[:, 1]
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'), row=(int(n / 3) + 1), col=(n % 3) + 1)
    
    fig.show()


def get_random_paths(centers_list, env_ids_list, qtt, folder):
    paths = []
    for center, env_id in zip(centers_list, env_ids_list):
        num_paths = len(os.listdir(f"{project_path}/{folder}/e{env_id}"))
        path_ids = np.random.choice(num_paths, qtt, replace=True)
        env_paths = []
        for path_id in path_ids:
            path = np.fromfile(f"{project_path}/{folder}/e{env_id}/path{path_id}.dat").reshape((-1, 2))
            env_paths.append(path)
        paths.append(env_paths)
    paths = np.array(paths)
    return paths


def plot_pnet_result(centers, paths, enet, pnet):
    enet = ContractiveAutoEncoder.load_from_checkpoint(enet)
    pnet = PNet.load_from_checkpoint(pnet)
    
    env_sample = torch.from_numpy(create_samples(centers))
    embedding = enet(env_sample)
    for target_path in paths[0]:
        pnet_input = torch.as_tensor([*embedding, *target_path[0], *target_path[-1]])


def main(args):
    if args.plot == "sample":
        centers = np.loadtxt(f"{project_path}/obs/perm.csv", delimiter=',')[:100].reshape((-1, 7, 2))
        env_ids = np.random.choice(100, 5, replace=True)
        centers = centers[env_ids]
        
        paths_list = get_random_paths(centers, env_ids, 5)
        plot_sample(centers, paths_list)
    else:
        centers = np.loadtxt(f"{project_path}/obs/perm.csv", delimiter=',')[0:110].reshape((-1, 7, 2))
        env_ids = np.random.choice(110, 1, replace=True)
        centers = centers[env_ids]
        paths_list = get_random_paths(centers, env_ids, 5, "valEnv")
        plot_pnet_result(centers, paths_list, args.enet, args.pnet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnet', default="", type=str)
    parser.add_argument('--enet', default="", type=str)
    parser.add_argument('--plot', default="sample", type=str)
    args = parser.parse_args()
    
    main(args)
