import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module


def main(args):
    os.environ["WANDB_CACHE_DIR"] = "/pscratch/sd/r/rmastand/.cache/wandb/"

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="extraction", save_dir = f"{args.data_dir}/extraction/", log_model="all")
            logger.experiment.config.update(args)
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(args.data_dir, name=args.classifier)

        checkpoint_loss = ModelCheckpoint(dirpath = f"{args.data_dir}/extraction/best_models/", filename="val_loss", monitor="loss/val", mode="min", verbose=1, auto_insert_metric_name=True)
        lr_monitor = LearningRateMonitor(logging_interval='step')

    

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            devices="auto",
            accelerator="cuda",
            deterministic=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            callbacks = [checkpoint_loss, lr_monitor],
            precision=args.precision,
            default_root_dir=f"{args.data_dir}/extraction/"
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)

        model.len_train_dataloader = len(data.train_dataloader())

        #if bool(args.pretrained):
       #     state_dict = os.path.join(
        #        "cifar10_models", "state_dicts", args.classifier + ".pt"
        #    )
        #    model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, data.test_dataloader())
        else:
            trainer.fit(model, data)
            #trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/global/cfs/cdirs/m3246/rmastand/polymathic/cifar10_prenorm/")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument( "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--mlp", type=str, default='1024-1024-1024')

    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--sim-coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')

    args = parser.parse_args()
    main(args)
