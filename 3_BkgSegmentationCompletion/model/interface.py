import pytorch_lightning as pl
import os
import torch

class LitModel(pl.LightningModule):

    def __init__(self, args, config_file):
        super().__init__()

        self.args = args
        self.config_file = config_file
        self.logdir = os.path.join(args.basedir, args.exp_name)
        self.create_model()
        self.save_hyperparameters()

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def create_model(self):
        raise NotImplemented("Implement the [create_model] function")

    def training_step(self):
        raise NotImplemented("Implement the [training_step] function")