import argparse
import os

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

import MPNet.enet.data_loader as ae_dl

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class PathToTar:
    @staticmethod
    def _load_envs(folder, qtt):
        envs = ae_dl.load_perms(qtt, 0)
        env_names = os.listdir(f"{project_path}/{folder}")
        env_names = [name for name in env_names if int(name.split('e')[-1]) < qtt]
        
        return envs, env_names
    
    @staticmethod
    def _get_path_files(folder, env_name, qtt):
        path_files = os.listdir(f"{project_path}/{folder}/{env_name}")
        path_files = [file for file in path_files if int(file.split("path")[-1].split('.')[0]) < qtt]
        return path_files
    
    @staticmethod
    def _join_path_files(envs, folder, env_names, paths_per_env, output_path):
        key_id = 0
        sink = wds.TarWriter(output_path, encoder=False, compress=True)
        for env, env_name in zip(envs, env_names):
            env = ae_dl.create_samples(env)
            path_files = PathToTar._get_path_files(folder, env_name, paths_per_env)
            for file in path_files:
                path = np.fromfile(f"{project_path}/{folder}/{env_name}/{file}", dtype=float).reshape((-1, 2))
                for n, point in enumerate(path[:-1]):
                    sample = {"__key__":        f"data_{key_id}",
                              "env.ten":        env.tobytes(),
                              "trajectory.ten": np.array([*point, *path[-1]]).tobytes(),
                              "target.ten":     np.array([*path[n + 1]]).tobytes()}
                    
                    sink.write(sample)
                    key_id += 1
        sink.close()
    
    @staticmethod
    def to_tar(parent_folder, num_envs, paths_per_env, output_path):
        loaded_envs, env_names = PathToTar._load_envs(parent_folder, num_envs)
        PathToTar._join_path_files(loaded_envs, parent_folder, env_names, paths_per_env, output_path)


class FromTar:
    def __init__(self, ae=None):
        self._ae = ae
    
    def decode(self, sample):
        inp = torch.from_numpy(np.frombuffer(sample['trajectory.ten']).reshape((-1,)))
        target = torch.from_numpy(np.frombuffer(sample['target.ten']))
        env = torch.from_numpy(np.frombuffer(sample['env.ten'])).float()
        env = self._ae(env)
        
        resulting_input = torch.cat([env, inp])
        return resulting_input, target
    
    def load_dataset(self, file_path, batch_size=1, num_workers=0, shuffle=0, ae=None):
        self._ae = ae
        dataset = wds.Dataset(file_path)
        dataset = wds.Processor(dataset, wds.map, self.decode)
        if shuffle:
            dataset = dataset.shuffle(shuffle)
        dataset = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--parent', type=str, default="")
    parser.add_argument('--num_envs', type=int, default=100)
    parser.add_argument('--paths_per_env', type=int, default=4000)
    parser.add_argument('--output_path', type=str, default='')
    
    args = parser.parse_args()
    
    PathToTar.to_tar(args.parent, args.num_envs, args.paths_per_env, args.output_path)
