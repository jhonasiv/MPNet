import torch
import torch.utils.data as data
import os
import numpy as np
from plotly.subplots import make_subplots
from torch.autograd import Variable
import torch.nn as nn
from plotly import graph_objs as go
import math
from AE.data_loader import loader

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

# Environment Encoder
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(2800, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(),
#                                      nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x
#
#
# # N=number of environments; NP=Number of Paths
# def load_dataset(N=100, NP=4000):
#     Q = Encoder()
#     Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
#     if torch.cuda.is_available():
#         Q.cuda()
#
#     obs_rep = np.zeros((N, 28), dtype=np.float32)
#     for i in range(0, N):
#         # load obstacle point cloud
#         temp = np.fromfile('../../dataset/obs_cloud/obc' + str(i) + '.dat')
#         temp = temp.reshape(len(temp) / 2, 2)
#         obstacles = np.zeros((1, 2800), dtype=np.float32)
#         obstacles[0] = temp.flatten()
#         inp = torch.from_numpy(obstacles)
#         inp = Variable(inp).cuda()
#         output = Q(inp)
#         output = output.data.cpu()
#         obs_rep[i] = output.numpy()
#
#     ## calculating length of the longest trajectory
#     max_length = 0
#     path_lengths = np.zeros((N, NP), dtype=np.int8)
#     for i in range(0, N):
#         for j in range(0, NP):
#             fname = '../../dataset/e' + str(i) + '/path' + str(j) + '.dat'
#             if os.path.isfile(fname):
#                 path = np.fromfile(fname)
#                 path = path.reshape(len(path) / 2, 2)
#                 path_lengths[i][j] = len(path)
#                 if len(path) > max_length:
#                     max_length = len(path)
#
#     paths = np.zeros((N, NP, max_length, 2), dtype=np.float32)  ## padded paths
#
#     for i in range(0, N):
#         for j in range(0, NP):
#             fname = '../../dataset/e' + str(i) + '/path' + str(j) + '.dat'
#             if os.path.isfile(fname):
#                 path = np.fromfile(fname)
#                 path = path.reshape(len(path) / 2, 2)
#                 for k in range(0, len(path)):
#                     paths[i][j][k] = path[k]
#
#     dataset = []
#     targets = []
#     for i in range(0, N):
#         for j in range(0, NP):
#             if path_lengths[i][j] > 0:
#                 for m in range(0, path_lengths[i][j] - 1):
#                     data = np.zeros(32, dtype=np.float32)
#                     for k in range(0, 28):
#                         data[k] = obs_rep[i][k]
#                     data[28] = paths[i][j][m][0]
#                     data[29] = paths[i][j][m][1]
#                     data[30] = paths[i][j][path_lengths[i][j] - 1][0]
#                     data[31] = paths[i][j][path_lengths[i][j] - 1][1]
#
#                     targets.append(paths[i][j][m + 1])
#                     dataset.append(data)
#
#     data = zip(dataset, targets)
#     random.shuffle(data)
#     dataset, targets = zip(*data)
#     return np.asarray(dataset), np.asarray(targets)
#
#
# # N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
# # Unseen_environments==> N=10, NP=2000,s=100, sp=0
# # seen_environments==> N=100, NP=200,s=0, sp=4000
# def load_test_dataset(N=100, NP=200, s=0, sp=4000):
#     obc = np.zeros((N, 7, 2), dtype=np.float32)
#     temp = np.fromfile('../../dataset/obs.dat')
#     obs = temp.reshape(len(temp) / 2, 2)
#
#     temp = np.fromfile('../../dataset/obs_perm2.dat', np.int32)
#     perm = temp.reshape(77520, 7)
#
#     ## loading obstacles
#     for i in range(0, N):
#         for j in range(0, 7):
#             for k in range(0, 2):
#                 obc[i][j][k] = obs[perm[i + s][j]][k]
#
#     Q = Encoder()
#     Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
#     if torch.cuda.is_available():
#         Q.cuda()
#
#     obs_rep = np.zeros((N, 28), dtype=np.float32)
#     k = 0
#     for i in range(s, s + N):
#         temp = np.fromfile('../../dataset/obs_cloud/obc' + str(i) + '.dat')
#         temp = temp.reshape(len(temp) / 2, 2)
#         obstacles = np.zeros((1, 2800), dtype=np.float32)
#         obstacles[0] = temp.flatten()
#         inp = torch.from_numpy(obstacles)
#         inp = Variable(inp).cuda()
#         output = Q(inp)
#         output = output.data.cpu()
#         obs_rep[k] = output.numpy()
#         k = k + 1
#     ## calculating length of the longest trajectory
#     max_length = 0
#     path_lengths = np.zeros((N, NP), dtype=np.int8)
#     for i in range(0, N):
#         for j in range(0, NP):
#             fname = '../../dataset/e' + str(i + s) + '/path' + str(j + sp) + '.dat'
#             if os.path.isfile(fname):
#                 path = np.fromfile(fname)
#                 path = path.reshape(len(path) / 2, 2)
#                 path_lengths[i][j] = len(path)
#                 if len(path) > max_length:
#                     max_length = len(path)
#
#     paths = np.zeros((N, NP, max_length, 2), dtype=np.float32)  ## padded paths
#
#     for i in range(0, N):
#         for j in range(0, NP):
#             fname = '../../dataset/e' + str(i + s) + '/path' + str(j + sp) + '.dat'
#             if os.path.isfile(fname):
#                 path = np.fromfile(fname)
#                 path = path.reshape(len(path) / 2, 2)
#                 for k in range(0, len(path)):
#                     paths[i][j][k] = path[k]
#
#     return obc, obs_rep, paths, path_lengths
