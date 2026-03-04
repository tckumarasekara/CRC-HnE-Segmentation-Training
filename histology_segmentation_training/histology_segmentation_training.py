import os
from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print
import torch
from data_loading.data_loader import ConicDataModule
import models.unet_instance
from models.unet_super import UnetSuper
from mlf_core.mlf_core import MLFCore

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = ArgumentParser()
    parser.add_argument(
        '--general-seed',
        type=int,
        default=0,
        help='General random seed',
    )
    parser.add_argument(
        '--pytorch-seed',
        type=int,
        default=0,
    help='Random seed of all Pytorch functions',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='log interval of stdout',
    )
    parser.add_argument(
        '--download',
        type=bool,
        default=False,
        help='If the data should be downloaded from Zenodo'
    )
    parser.add_argument(
        '--from-source',
        type=bool,
        default=False,
        help='Downloads the Conic data and applies necessary changes'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help="Training duration"
    )
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = UnetSuper.add_model_specific_args(parent_parser=parser)
    mlflow.autolog(True)
    # log conda env and system information
    try:
        MLFCore.log_sys_intel_conda_env()
    except:
        print("logging conda environment did not work")
    # parse cli arguments
    args = parser.parse_args()
    dict_args = vars(args)
    # store seed
    # number of gpus to make linter bit less restrict in terms of naming
    general_seed = dict_args['general_seed']
    pytorch_seed = dict_args['pytorch_seed']
    dict_args["max_epochs"] = dict_args["epochs"]
    num_of_gpus = torch.cuda.device_count()

    MLFCore.set_general_random_seeds(general_seed)
    MLFCore.set_pytorch_random_seeds(pytorch_seed, num_of_gpus)
    print(dict_args["accelerator"])
    if 'accelerator' in dict_args:
        if dict_args['accelerator'] == 'gpu':
            dict_args['accelerator'] = 'gpu'
        elif dict_args['accelerator'] == 'cpu':
            dict_args['accelerator'] = "cpu"
        else:
            print(
                f'[bold red]{dict_args["accelerator"]}[bold blue] currently not supported. Switching to [bold '
                f'green]cpu!')
            dict_args['accelerator'] = 'cpu'
    dm = ConicDataModule(**dict_args)
    dict_args["num_classes"] = 7
    MLFCore.log_input_data('histology_segmentation_training/data/OME-TIFFs/')
    dm.setup(stage='fit')
    model = models.unet_instance.__getattr__(dict_args["models"])
    if torch.cuda.is_available():
        model = model(hparams=parser.parse_args(), input_channels=3, min_filter=32, on_gpu=True, **dict_args)
        model.cuda()
    else:
        model = model(hparams=parser.parse_args(), input_channels=3, min_filter=32, on_gpu=False, **dict_args)
    model.log_every_n_steps = dict_args['log_interval']

    # check, whether the run is inside a Docker container or not
    if 'MLF_CORE_DOCKER_RUN' in os.environ:
        checkpoint_callback = ModelCheckpoint(filename="seg_training_main/mlruns/ckpt", save_top_k=0, verbose=True,
                                              monitor='val_mean_iou', mode='max')
        trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, default_root_dir='/data',
                                                logger=TensorBoardLogger('/data'))
        tensorboard_output_path = f'data/default/version_{trainer.logger.version}'
    else:
        checkpoint_callback = ModelCheckpoint(filename=f'{os.getcwd()}/mlruns/best', save_top_k=1,
                                              verbose=True, monitor='val_mean_iou', mode='max')
        if torch.cuda.is_available():
            trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback],
                                                    default_root_dir=os.getcwd() + "/mlruns",
                                                    logger=TensorBoardLogger('out'),
                                                    gpus=[0])
        else:
            trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback],
                                                    default_root_dir=os.getcwd() + "/mlruns",
                                                    logger=TensorBoardLogger('out'))
        tensorboard_output_path = f'data/default/version_{trainer.logger.version}'

    trainer.deterministic = True
    trainer.benchmark = False
    trainer.log_every_n_steps = dict_args['log_interval']
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    print(f'\n[bold blue]For tensorboard log, call [bold green]tensorboard --logdir={tensorboard_output_path}')
