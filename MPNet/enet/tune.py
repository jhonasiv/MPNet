from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from CAE import ContractiveAutoEncoder
import pytorch_lightning as pl
import data_loader as dl
import argparse
from torch import nn


def train(config, batch_size, num_epochs=20, num_gpus=0):
    training = dl.loader(55000, batch_size, 0)
    validation = dl.loader(8250, 1, 55000)
    cae = ContractiveAutoEncoder(training_dataloader=training, val_dataloader=validation, config=config)
    trainer = pl.Trainer(max_epochs=num_epochs, gpus=num_gpus, auto_select_gpus=True if num_gpus else False,
                         logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version='.'),
                         stochastic_weight_avg=True, benchmark=True,
                         callbacks=[TuneReportCheckpointCallback({"loss": "val_loss"},
                                                                 filename="checkpoint", on="validation_end")])
    
    trainer.fit(cae)


def tuning(args):
    activation = nn.PReLU if args.actv == 'prelu' else nn.SELU
    config = {"l1_units": tune.choice([480, 512, 544]),
              "l2_units": tune.choice([224, 256, 288]),
              "l3_units": tune.choice([96, 128, 160]),
              "lambda":   tune.choice([1e-3, 1e-4, 1e-5]),
              "actv":     tune.choice([activation])}
    scheduler = PopulationBasedTraining(time_attr='training_iteration', perturbation_interval=4,
                                        hyperparam_mutations={"l1_units": [464, 496, 528, 560, 576],
                                                              "l2_units": [208, 240, 272, 304, 328],
                                                              "l3_units": [80, 112, 144, 176, 208]})
    
    reporter = CLIReporter(parameter_columns=["l1_units", "l2_units", "l3_units", "lambda", ],
                           metric_columns=["loss", "training_iteration"])
    
    analysis = tune.run(tune.with_parameters(train, batch_size=args.batch_size, num_epochs=args.num_epochs,
                                             num_gpus=args.num_gpus),
                        resources_per_trial={"cpu": args.num_cpus, "gpu": args.num_gpus}, metric="loss",
                        mode="min", config=config, num_samples=args.num_trials, scheduler=scheduler,
                        progress_reporter=reporter, max_failures=3, stop={"training_iteration": 10},
                        name="tune_cae")
    
    print(f"Found best hyperparameters: {analysis.best_config}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--num_cpus', default=1, type=int)
    parser.add_argument('--num_trials', default=10, type=int)
    parser.add_argument('--actv', default='prelu', type=str)
    parser.add_argument('--resume', const=True, nargs="?")
    
    args = parser.parse_args()
    
    tuning(args)
