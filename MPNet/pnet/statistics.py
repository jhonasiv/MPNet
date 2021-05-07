import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adagrad, AdamW

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.enet.statistics import TrainingDataCallback
from MPNet.pnet.model import PNet
from data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def train(args):
    configs = [{"dropout_rate": 0.5, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": 1e-2},
               {"dropout_rate": 0.4, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": 1e-2},
               {"dropout_rate": 0.3, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": 1e-2},
    
    prelu_cae = ContractiveAutoEncoder.load_from_checkpoint(f"{project_path}/models/cae.ckpt")
    suggested_cae = ContractiveAutoEncoder.load_from_checkpoint(f"{project_path}/models/cae_prelu_2.pt")
    
    enet_implementations = [prelu_cae, suggested_cae]
    
    enet_suffix = "qenet"
    for enet in enet_implementations:
        training = loader(enet, f"{project_path}/env", 100, 0, args.batch_size)
        validation = loader(enet, f"{project_path}/valEnv", 110, 0, args.batch_size)
        for n, config in enumerate(configs):
            for itt in range(args.itt):
                es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
                logging = TrainingDataCallback(f"{args.log_path}/pnet_{n}_{itt}_{enet_suffix}.json",
                                               log_stats=["val_loss", "epoch"])
                
                trainer = pl.Trainer(gpus=args.num_gpus, callbacks=[es, logging], deterministic=True)
                pnet = PNet(32, 2, training_dataloader=training, validation_dataloader=validation, config=config,
                            reduce=True)
                
                trainer.fit(pnet)
        enet_suffix = "newenet"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=250, type=int)
    parser.add_argument('--itt', default=20, type=int)
    parser.add_argument('--log_path', default="", type=str)
    parser.add_argument('--num_gpus', default=int, type=0)
    args = parser.parse_args()
