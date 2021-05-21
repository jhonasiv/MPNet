import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim import Adagrad, AdamW

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.enet.statistics import TrainingDataCallback
from MPNet.pnet.model import PNet
from data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def train(args):
    configs = [{"dropout_rate": 0.5, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr},
               {"dropout_rate": 0.4, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr},
               {"dropout_rate": 0.3, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr},
               {"dropout_rate": 0.15, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr},
               {"dropout_rate": 0, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr}]
    
    config = configs[args.model_id]
    for enet in args.enet_models:
        training_config = {"enet"       : f"{project_path}/{enet}", "paths_folder": f"{project_path}/env",
                           "qtt_envs"   : 100, "envs_start_idx": 0, "batch_size": args.batch_size, "shuffle": True,
                           "num_workers": args.workers}
        
        validation_config = {"enet"       : f"{project_path}/{enet}", "paths_folder": f"{project_path}/valEnv",
                             "qtt_envs"   : 110, "envs_start_idx": 0, "batch_size": args.batch_size, "shuffle": False,
                             "num_workers": args.workers};
        enet_suffix = os.path.basename(enet).split('.')[0]
        es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
        logging = TrainingDataCallback(f"{args.log_path}/pnet_{args.model_id}_{enet_suffix}.json",
                                       log_stats=["val_loss", "epoch"])
        checkpointing = ModelCheckpoint(monitor='val_loss', dirpath=f"{project_path}/{args.model_output_path}",
                                        filename=f"pnet_{args.model_id}_{enet_suffix}", verbose=True, save_top_k=1)
        
        if args.resume:
            trainer = pl.Trainer(callbacks=[es, logging, checkpointing], max_epochs=args.num_epochs,
                                 resume_from_checkpoint=f"{project_path}/{args.model_output_path}/pnet_"
                                                        f"{args.model_id}_{enet_suffix}.ckpt", gpus=1)
        else:
            trainer = pl.Trainer(callbacks=[es, logging, checkpointing], max_epochs=args.num_epochs, gpus=1)
        pnet = PNet(32, 2, config=config, training_config=training_config, validation_config=validation_config,
                    reduce=True)
        
        trainer.fit(pnet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=250, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument('--log_path', default=".", type=str)
    parser.add_argument('--num_gpus', default=0, type=int)
    parser.add_argument("--enet_models", default=[], nargs="+", required=True)
    parser.add_argument("--model_output_path", default="", type=str, help="Output path for model without extension")
    parser.add_argument("--model_id", default=0, type=int, required=True)
    parser.add_argument("--num_epochs", default=500, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--resume", nargs='?', type=bool, const=True)
    args = parser.parse_args()
    
    train(args)
