import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.pnet.model import PNet
from data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def train(args):
    if not os.path.exists(f"{project_path}/{args.model_path}"):
        os.makedirs(f'{project_path}/{args.model_path}')
    
    pl.seed_everything(42)
    
    cae = ContractiveAutoEncoder.load_from_checkpoint(args.enet)
    
    training = loader(cae, f"{project_path}/env", 100, 0, args.batch_size, num_workers=2, persistent_workers=True)
    validation = loader(cae, f"{project_path}/valEnv", 110, 0, args.batch_size, num_workers=2)
    
    pnet = PNet(32, 2, training_dataloader=training, validation_dataloader=validation, config={'linear': False},
                reduce=True)
    
    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, mode='min', verbose=True)
    checkpointing = ModelCheckpoint(monitor='val_loss', dirpath=f"{project_path}/{args.model_path}/",
                                    filename=args.output_filename, verbose=True, save_top_k=1)
    trainer = pl.Trainer(gpus=1, benchmark=False, deterministic=True, callbacks=[checkpointing],
                         max_epochs=args.num_epochs, profiler=Profiler)
    
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
