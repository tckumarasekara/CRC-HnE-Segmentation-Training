import os
from argparse import ArgumentParser

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.autolog(True)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print
import torch
from data_loading.data_loader import ConicDataModule, ConicData
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

    parser = UnetSuper.add_model_specific_args(parent_parser=parser)

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

    # load data
    dm = ConicDataModule(**dict_args)
    dict_args["num_classes"] = 7
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/OME-TIFFs/')
    MLFCore.log_input_data(img_path)

    # prepare data
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
                                              monitor='train_mean_iou', mode='min')
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=dict_args["epochs"],
            callbacks=[checkpoint_callback],
            default_root_dir="/data",
            logger=TensorBoardLogger("/data"),
            log_every_n_steps=dict_args["log_interval"],
            deterministic=True,
        )
        tensorboard_output_path = f'data/default/version_{trainer.logger.version}'

    else:
        checkpoint_callback = ModelCheckpoint(filename=f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/mlruns/models/best_{dict_args["models"]}_lr-{dict_args["lr"]}_wd-{dict_args["weight_decay"]}_dropout-{dict_args["dropout_val"]}_epoch-{dict_args["epochs"]}_batchS-{dict_args["training_batch_size"]}',
                                              save_top_k=1, verbose=True, monitor='val_mean_iou', mode='max')
        if torch.cuda.is_available():
            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=dict_args["epochs"],
                callbacks=[checkpoint_callback],
                default_root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/mlruns/models",
                logger=TensorBoardLogger('out'),
                log_every_n_steps=dict_args["log_interval"],
                deterministic=True,
            )
        else:
            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=dict_args["epochs"],
                callbacks=[checkpoint_callback],
                default_root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/mlruns/models",
                logger=TensorBoardLogger('out'),
                log_every_n_steps=dict_args["log_interval"],
                deterministic=True,
            )
        tensorboard_output_path = f'out/lightning_logs/version_{trainer.logger.version}'

    trainer.deterministic = True
    trainer.benchmark = False
    trainer.log_every_n_steps = dict_args['log_interval']
    trainer.fit(model, dm)
    trainer.test(datamodule=dm, ckpt_path='best')
    print(f'\n[bold blue]For tensorboard log, call [bold green]tensorboard --logdir={tensorboard_output_path}')
