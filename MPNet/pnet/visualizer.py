import torch
import torch.utils.data as data
import os
import numpy as np
from plotly.subplots import make_subplots
from torch.autograd import Variable
import torch.nn as nn
from plotly import graph_objs as go
import math
from enet.data_loader import loader

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


if __name__ == '__main__':
    centers = np.loadtxt(f"{project_path}/obs/perm.csv", delimiter=',')[:100].reshape((-1, 7, 2))
    env_ids = np.random.choice(100, 5, replace=True)
    centers = centers[env_ids]
    
    paths_list = get_random_paths(centers, env_ids, 5)
    plot_sample(centers, paths_list)
