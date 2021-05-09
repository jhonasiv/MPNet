import argparse
import os
from abc import ABC

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adagrad, AdamW

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.enet.statistics import TrainingDataCallback
from MPNet.pnet.model import PNet
from data_loader import loader
from google.cloud import storage

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class GoogleCloudCheckpoint(pl.callbacks.ModelCheckpoint, ABC):
    def __init__(self, bucket, *args, **kwargs):
        super(GoogleCloudCheckpoint, self).__init__(*args, **kwargs)


def train(args):
    client = storage.Client(args.project)
    bucket = client.get_bucket(args.bucket)
    
    device = {"tpu_cores": args.num_tpus} if args.num_tpus > 0 else {"gpus": args.num_gpus}
    
    configs = [{"dropout_rate": 0.5, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr},
               {"dropout_rate": 0.4, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr},
               {"dropout_rate": 0.3, "activation": nn.PReLU, 'optimizer': Adagrad, "lr": args.lr}, ]
    
    config = configs[args.model_id]
    for enet in args.enet_models:
        training_config = {"enet"      : enet, "paths_folder": "env", "qtt_envs": 100, "envs_start_idx": 0,
                           "batch_size": args.batch_size, "shuffle": True, "project": args.project,
                           "bucket"    : args.bucket, "path": "obs/perm.csv"}
        
        validation_config = {"enet"      : enet, "paths_folder": "valEnv", "qtt_envs": 110, "envs_start_idx": 0,
                             "batch_size": args.batch_size, "shuffle": False, "project": args.project,
                             "bucket"    : args.bucket, "path": "obs/perm.csv"}
        enet_suffix = os.path.basename(enet).split('.')[0]
        es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
        logging = TrainingDataCallback(args.project, args.bucket,
                                       f"{args.log_path}/pnet_{args.model_id}_{enet_suffix}.json",
                                       log_stats=["val_loss", "epoch"])
        checkpointing = ModelCheckpoint(monitor='val_loss', dirpath=f"{project_path}/{args.model_output_path}",
                                        filename=args.output_filename, verbose=True, save_top_k=1)
        
        trainer = pl.Trainer(callbacks=[es, logging, checkpointing], max_epochs=args.num_epochs, **device)
        pnet = PNet(32, 2, config=config, training_config=training_config, validation_config=validation_config,
                    reduce=True)
        
        trainer.fit(pnet)
        blob = bucket.blob(f"{args.model_output_path}_{args.model_id}_{enet_suffix}.pt")
        state_dict = pnet.state_dict()
        state_dict['config'] = config
        blob.upload_from_string(json.dumps(pnet.state_dict()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", default="", type=str)
    parser.add_argument('--batch-size', default=250, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument('--log_path', default=".", type=str)
    parser.add_argument('--num_gpus', default=0, type=int)
    parser.add_argument("--num_tpus", default=0, type=int)
    parser.add_argument("--project", default="", type=str, help="Google cloud project ID")
    parser.add_argument("--bucket", default="", type=str, help="Google cloud storage bucket")
    parser.add_argument("--enet_models", default=[], nargs="+", required=True)
    parser.add_argument("--model_output_path", default="", type=str, help="Output path for model without extension")
    parser.add_argument("--model_id", default=0, type=int, required=True)
    parser.add_argument("--num_epochs", default=500, type=int)
    args = parser.parse_args()
    
    train(args)
