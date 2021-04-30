import os
from abc import ABC

import numpy as np
import torch
from torch.utils.data import IterableDataset

import AE.data_loader as ae_dl
from AE.CAE import ContractiveAutoEncoder

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def load_envs(folder, qtt):
    envs = ae_dl.load_perms(qtt, 0)
    env_names = os.listdir(f"{project_path}/{folder}")
    env_names = [name for name in env_names if int(name.split('e')[-1]) < qtt]
    
    return envs, env_names


def get_path_files(folder, env_name, qtt):
    path_files = os.listdir(f"{project_path}/{folder}/{env_name}")
    path_files = [file for file in path_files if int(file.split("path")[-1].split('.')[0]) < qtt]
    return path_files


def join_path_files(folder, env_names, paths_per_env, filename):
    data = []
    for env_name in env_names:
        path_files = get_path_files(folder, env_name, paths_per_env)
        for file in path_files:
            path = np.fromfile(f"{project_path}/{folder}/{env_name}/{file}", dtype=float).reshape((-1, 2))
            for n, point in enumerate(path[:-1]):
                data.append([[env_name, point, path[-1]], [path[n + 1]]])
    data = np.array(data)
    np.save(f"{project_path}/processed/{filename}", data)


def process_data_files(envs, path_files, cae):
    sample = ae_dl.create_samples(envs[0], {})
    sample = torch.tensor(sample).float()
    cae = ContractiveAutoEncoder.load_from_checkpoint(f"{project_path}/models/cae.ckpt")
    embedding = cae(sample)
    a = 0


class MPNetDataLoader(IterableDataset, ABC):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    a = np.load(f"{project_path}/processed/training.npy", allow_pickle=True)
    env, names = load_envs('env', 100)
    path_names = []
    join_path_files('env', names, 4000, 'training')
