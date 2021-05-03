import argparse
import os

import pytorch_lightning as pl
import sys
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.pnet.model import PNet
from data_loader import FromTar

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"

sys.setrecursionlimit(1000)


def train(args):
    if not os.path.exists(f"{project_path}/{args.model_path}"):
        os.makedirs(f'{project_path}/{args.model_path}')
    
    pl.seed_everything(42)
    
    cae = ContractiveAutoEncoder.load_from_checkpoint(args.enet)
    tar_loader = FromTar()
    training = tar_loader.load_dataset(f"{project_path}/datasets/training.tar", args.batch_size, shuffle=5000,
                                       ae=cae)
    validation = tar_loader.load_dataset(f"{project_path}/datasets/validation.tar", 1, ae=cae)
    
    pnet = PNet(training_dataloader=training, validation_dataloader=validation, reduce=True)
    
    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='min', verbose=True)
    checkpointing = ModelCheckpoint(monitor='val_loss', dirpath=f"{project_path}/{args.model_path}/",
                                    filename=args.output_filename, verbose=True, save_top_k=1)
    trainer = pl.Trainer(gpus=1, callbacks=[es, checkpointing], deterministic=True,
                         benchmark=True, progress_bar_refresh_rate=1)
    
    trainer.fit(pnet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--enet', default="", type=str, required=True)
    
    parser.add_argument('--output_filename', default="pnet", type=str)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
