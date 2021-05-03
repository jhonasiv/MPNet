import argparse
import os

import pytorch_lightning as pl

from MPNet.enet.CAE import ContractiveAutoEncoder
from data_loader import FromTar

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def train(args):
    if not os.path.exists(f"{project_path}/{args.model_path}"):
        os.makedirs(f'{project_path}/{args.model_path}')
    
    pl.seed_everything(42)
    
    cae = ContractiveAutoEncoder.load_from_checkpoint(args.enet)
    tar_loader = FromTar()
    training = tar_loader.load_dataset(f"{project_path}/datasets/training.tar", 250, shuffle=5000, ae=cae)
    validation = tar_loader.load_dataset(cae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--enet', default="", type=str, required=True)
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--learning_rate', type=float, default=1 - 4)
    args = parser.parse_args()
    
    train(args)
