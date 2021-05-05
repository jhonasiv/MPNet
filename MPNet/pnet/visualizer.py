import argparse
import os

import numpy as np
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.enet.data_loader import create_samples
from MPNet.pnet.model import PNet

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def plot_sample(center_list, path_list):
    fig = make_subplots(rows=2, cols=3)
    for n, (center, paths) in enumerate(zip(center_list, path_list)):
        obstacles = []
        for point in center:
            x, y = point
            obstacles.extend([[x - 2.5, y - 2.5], [x - 2.5, y + 2.5], [x + 2.5, y + 2.5], [x + 2.5, y - 2.5],
                              [x - 2.5, y - 2.5], [None, None]])
        obstacles = np.array(obstacles)
        x = obstacles[:, 0]
        y = obstacles[:, 1]
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor="black", name="obstacle"),
                      row=(int(n / 3) + 1), col=(n % 3) + 1)
        
        for path in paths:
            x = path[:, 0]
            y = path[:, 1]
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'), row=(int(n / 3) + 1), col=(n % 3) + 1)
    
    fig.show()


def get_random_paths(centers_list, env_ids_list, qtt):
    paths = []
    for center, env_id in zip(centers_list, env_ids_list):
        path_ids = np.random.choice(4000, qtt, replace=True)
        env_paths = []
        for path_id in path_ids:
            path = np.fromfile(f"{project_path}/env/e{env_id}/path{path_id}.dat").reshape((-1, 2))
            env_paths.append(path)
        paths.append(env_paths)
    paths = np.array(paths)
    return paths


def plot_pnet_result(centers, enet, pnet):
    enet = ContractiveAutoEncoder.load_from_checkpoint(enet)
    pnet = PNet.load_from_checkpoint(pnet)
    
    env_id = np.random.choice(110, 1)
    
    env_sample = create_samples(centers[env_id])
    embedding = enet(env_sample)


def main(args):
    if args.plot == "sample":
        centers = np.loadtxt(f"{project_path}/obs/perm.csv", delimiter=',')[:100].reshape((-1, 7, 2))
        env_ids = np.random.choice(100, 5, replace=True)
        centers = centers[env_ids]
        
        paths_list = get_random_paths(centers, env_ids, 5)
        plot_sample(centers, paths_list)
    else:
        centers = np.loadtxt(f"{project_path}/obs/perm.csv", delimiter=',')[0:110].reshape((-1, 7, 2))
        plot_pnet_result(centers, args.enet, args.pnet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnet', default="", type=str)
    parser.add_argument('--enet', default="", type=str)
    parser.add_argument('--plot', default="sample", type=str)
    args = parser.parse_args()
    
    main(args)
