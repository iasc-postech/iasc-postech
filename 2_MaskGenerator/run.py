from gettext import find
import config
import os
import yaml

from select_option import select_model, select_dataset

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin, DDPShardedPlugin

import logging


if __name__ == "__main__":
    
    logging.getLogger("lightning").setLevel(logging.ERROR)
    args, parser = config.config_parser()

    with open(args.config, "r") as fp: 
        config_file = yaml.load(fp, Loader=yaml.FullLoader)

    config_file["coco_path"] = args.coco_path
    config_file["modified_coco_path"] = args.modified_coco_path

    model_name = config_file["model"]
    
    dataset_name = config_file["dataset_name"]
    model_fn = select_model(model_name)
        
    dataset = select_dataset(dataset_name=dataset_name, config_file=config_file)
    basedir = args.basedir

    if args.train:
        args.exp_name = "train" + "_" + model_name + "_" + config_file["exp_name"] 
    elif args.test:
        args.exp_name = "test" + "_" + model_name + "_" + config_file["exp_name"] 

    if args.debug:
        args.exp_name += "_debug"
    
    config_file["exp_name"] = args.exp_name
    logdir = os.path.join(basedir, args.exp_name)
    n_gpus = torch.cuda.device_count()
    os.makedirs(logdir, exist_ok=True)

    checkpoint_path = '{save_base_dir}/{exp_name}'.format(save_base_dir=config_file['save_base_dir'], exp_name=args.exp_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.train:
        f = os.path.join(logdir, "args.txt")
        with open(f, "w") as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write("{} = {}\n".format(arg, attr))

    project_name = config_file["model"]
    ## Wandb Logger Init
    wandb_logger = pl_loggers.WandbLogger(name=args.exp_name, project="iasc", save_dir=config_file['wandb_log_path'])

    ## Seed Fix
    seed_everything(args.seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model_checkpoint = ModelCheckpoint(
        monitor="val_fid",
        dirpath=checkpoint_path,
        filename='{epoch:02d}-{val_fid:.3f}',
        auto_insert_metric_name=True,            
        save_top_k=20,
        every_n_epochs=config_file['check_val_every_n_epoch'],
        mode="min",
        )

    callbacks = [lr_monitor, model_checkpoint]

    if config_file['strategy'] == 'ddp':
        strategy = DDPPlugin(gradient_as_bucket_view=True, find_unused_parameters=True)
    elif config_file['strategy'] == 'deepspeed':
        strategy = "deepspeed_stage_3_offload"

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=args.i_print,
        devices=n_gpus,
        max_epochs=-1,
        accelerator="gpu",
        #replace_sampler_ddp=False,
        sync_batchnorm = True,
        # accumulate_grad_batches = config_file['accumulation_numb'],
        # deterministic=True,
        # strategy = "ddp_sharded",
        # strategy = "deepspeed_stage_3_offload",
        strategy = strategy,
        check_val_every_n_epoch=config_file['check_val_every_n_epoch'],
        # val_check_interval=config_file['val_check_interval'],
        precision = config_file['precision'],
        num_sanity_val_steps=2,
        # limit_train_batches=5,# limit_val_batches=5, ## For Test
        callbacks=callbacks,
        profiler="simple",
        # gradient_clip_val = 0.5
        )

        
    model = model_fn(args, config_file)
    if args.train:
        if config_file['start_from_checkpoint'] == True:
            trainer.fit(model, dataset, ckpt_path=config_file['checkpoint_path'])
        else:
            trainer.fit(model, dataset)
    elif args.test:
        trainer.test(model, dataset, ckpt_path=config_file['checkpoint_path'])
    else:
        print("check the argument")