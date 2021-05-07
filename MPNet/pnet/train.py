import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW, Adagrad
from torch import nn

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.pnet.model import PNet
from MPNet.pnet.data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def train(args):
    cae = ContractiveAutoEncoder.load_from_checkpoint(args.enet)
    cae.freeze()
    # training = loader(cae, f"{project_path}/env", 100, 0, args.batch_size, num_workers=4, shuffle=True)
    # validation = loader(cae, f"{project_path}/valEnv", 110, 0, args.batch_size * 2, num_workers=4, shuffle=False)
    
    # pnet = PNet(32, 2, training_dataloader=training, validation_dataloader=validation,
    #             config={'lr': args.learning_rate, 'optimizer': Adagrad, 'actv': nn.SELU}, reduce=True)
    
    pnet = PNet(32, 2, training_config={"enet"    : args.enet, "paths_folder": f"{project_path}/env",
                                        "qtt_envs": 100, "envs_start_idx": 0, "batch_size": args.batch_size,
                                        "shuffle" : True, "num_workers": 4},
                validation_config={"enet"    : args.enet, "paths_folder": f"{project_path}/valEnv",
                                   "qtt_envs": 110, "envs_start_idx": 0, "batch_size": args.batch_size,
                                   "shuffle" : False, "num_workers": 4},
                config={'lr': args.learning_rate, 'optimizer': Adagrad, 'actv': nn.SELU},
                reduce=True)
    
    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, mode='min', verbose=True)
    checkpointing = ModelCheckpoint(monitor='val_loss', dirpath=f"{project_path}/{args.model_path}/",
                                    filename=args.output_filename, verbose=True, save_top_k=3)
    trainer = pl.Trainer(gpus=1, benchmark=True, callbacks=[checkpointing, es], stochastic_weight_avg=True,
                         max_epochs=args.num_epochs)
    
    trainer.fit(pnet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--enet', default="", type=str, required=True)
    parser.add_argument('--output_filename', default="pnet", type=str)
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
