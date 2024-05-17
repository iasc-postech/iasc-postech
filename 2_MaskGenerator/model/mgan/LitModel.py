from cProfile import label
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as f


from collections import OrderedDict
from model.mgan.architecture import Generator, Discriminator, MODULES
from model.interface import LitModel
import utils.misc as misc
from utils.diffaug import apply_diffaug as diffaug
from utils.losses import d_hinge, g_hinge

import wandb

class MGAN(LitModel):
    def create_model(self):

        self.automatic_optimization = False
        self.D_steps_per_G = self.config_file['D_steps_per_G']
        self.mgan_config = self.config_file['mgan_config']
        self.num_classes = self.config_file['label_nc']
        self.batch_size = self.config_file['batch_size']
        self.class_list = self.config_file['category_id']
        module = MODULES(apply_g_sn=self.mgan_config['apply_g_sn'], apply_d_sn=self.mgan_config['apply_d_sn'])

        self.generator = Generator(
            z_dim=self.mgan_config['z_dim'],
            g_shared_dim=self.mgan_config['g_shared_dim'],
            img_size=self.config_file['crop_size'],
            g_conv_dim=self.mgan_config['g_conv_dim'],
            apply_attn=self.mgan_config['apply_attn'],
            attn_g_loc=self.mgan_config['attn_g_loc'],
            num_classes=self.num_classes,
            MODULES=module,
            extra_feature_condition=self.mgan_config['extra_feature_condition']
            )
        
        self.discriminator = Discriminator(
                img_size=self.config_file['crop_size'],
                d_conv_dim=self.mgan_config['d_conv_dim'],
                apply_d_sn = self.mgan_config['apply_d_sn'],
                apply_attn=self.mgan_config['apply_attn'],
                attn_d_loc=self.mgan_config['attn_d_loc'],
                num_classes=self.num_classes,
                MODULES=module,
                extra_feature_condition=self.mgan_config['extra_feature_condition']
                )
        
        self.fixed_fake_label = self.sample_fixed_fake_label()
        self.fixed_noise = torch.randn(self.fixed_fake_label.shape[0], self.generator.z_dim)
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.fid = FrechetInceptionDistance(feature=2048)


    def configure_optimizers(self):

        opt_d = torch.optim.Adam(params=self.discriminator.parameters(),
                                lr=self.config_file['d_lr_init'],
                                betas=(self.config_file['beta_1'], self.config_file['beta_2']))

        opt_g = torch.optim.Adam(params=self.generator.parameters(),
                                lr=self.config_file['g_lr_init'],
                                betas=(self.config_file['beta_1'], self.config_file['beta_2']))


        return [opt_d, opt_g], []

    def sample_fixed_fake_label(self):
        import math
        label = np.array([c for i in range(8) for c in self.class_list])
        return torch.tensor(label).int()

    def sample_fake_label(self):
        import math
        label = np.array([c for i in range(math.ceil(self.batch_size/len(self.class_list))) for c in self.class_list])
        label = np.random.permutation(label)
        label = label[:self.batch_size]
        return torch.tensor(label).int()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.total_loss = 0

        opt_d, opt_g = self.optimizers()
        
        image, label = batch
        basket_idx = 0

        try:
            if batch_idx % self.D_steps_per_G == 0:
                z = torch.randn(label.shape[0], self.generator.z_dim)
                z = z.type_as(image)
                fake_label = self.sample_fake_label()
                fake_label = fake_label.type_as(label)
                g_loss, g_step_fake_images, g_step_fake_images_aug = self.train_generator(z=z, fake_labels=fake_label, node_feature=None, real_labels=label, real_images_=image)
                self.manual_backward(g_loss)
                basket_idx += 1

                opt_g.step()
                opt_g.zero_grad()


            z = torch.randn(label.shape[0], self.generator.z_dim)
            z = z.type_as(image)
            fake_label = self.sample_fake_label()
            fake_label = fake_label.type_as(label)
            d_loss, d_step_fake_images, d_step_real_images, d_step_fake_images_aug, d_step_real_images_aug = self.train_discriminator(z=z, fake_labels=fake_label, node_feature=None, \
                real_labels=label, real_images_=image)
            self.manual_backward(d_loss)
            basket_idx += 1

            opt_d.step()
            opt_d.zero_grad()

        except IndexError:
            import pdb; pdb.set_trace()
        if batch_idx % self.D_steps_per_G == 0:
            self.log("train_{loss_name}".format(loss_name="dis_loss"), d_loss.item(), on_step=True, logger=True)
            self.log("train_{loss_name}".format(loss_name="gen_loss"), g_loss.item(), on_step=True, logger=True)

        if batch_idx % 40 == 0:
            import torchvision 
            z = torch.randn(len(self.class_list)*8, self.generator.z_dim)
            z = z.type_as(image)
            # fake_sample = self.make_sample(z, self.fixed_fake_label.type_as(label), torch.cat([feature, feature],dim=0)[:z.shape[0]])
            # fixed_fake_sample = self.make_sample(self.fixed_noise.type_as(image), self.fixed_fake_label.type_as(label), torch.cat([feature,feature],dim=0)[:z.shape[0]])
            fake_sample = self.make_sample(z, self.fixed_fake_label.type_as(label), None)
            fixed_fake_sample = self.make_sample(self.fixed_noise.type_as(image), self.fixed_fake_label.type_as(label), None)

            #https://pytorch.org/vision/stable/_modules/torchvision/utils.html#save_image
            grid_real = torchvision.utils.make_grid((d_step_real_images+1)/2, nrow=8).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            grid_fake = torchvision.utils.make_grid((fake_sample+1)/2, nrow=8).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            grid_fixed_fake = torchvision.utils.make_grid((fixed_fake_sample+1)/2, nrow=8).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            self.logger.experiment.log({
                "train/real_image": wandb.Image(grid_real)
            })

            self.logger.experiment.log({
                "train/fake_image": wandb.Image(grid_fake)
            })

            self.logger.experiment.log({
                "train/fixed_fake_image": wandb.Image(grid_fixed_fake)
            })

            if self.config_file['diffaug'] == True:
                grid_real_aug = torchvision.utils.make_grid((d_step_real_images_aug+1)/2).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                grid_fake_aug = torchvision.utils.make_grid((d_step_fake_images_aug+1)/2).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                self.logger.experiment.log({
                    "train/real_image_aug": wandb.Image(grid_real_aug)
                })
                self.logger.experiment.log({
                    "train/fake_image_aug": wandb.Image(grid_fake_aug)
                })


    def train_discriminator(self, z, fake_labels, node_feature, real_labels, real_images_):
        ## toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.generator, grad=False, num_freeze_layers=-1, is_stylegan=False)
        misc.toggle_grad(model=self.discriminator, grad=True, num_freeze_layers=-1, is_stylegan=False)
        # ## Untrack
        self.generator.apply(misc.untrack_bn_statistics)

        fake_images_ = self.generator(z=z, label=fake_labels, node_feature=node_feature)
        fake_images_ = fake_images_.detach()
        fake_images_aug, real_images_aug = None, None

        if self.config_file['diffaug'] == True:
            real_images_aug, fake_images_aug = diffaug(x=real_images_, policy='translation,cutout'), diffaug(x=fake_images_, policy='translation,cutout')
            real_adv_output = self.discriminator(real_images_aug, real_labels, node_feature=node_feature)
            fake_adv_output = self.discriminator(fake_images_aug, fake_labels, node_feature=node_feature)
        else:
            real_adv_output = self.discriminator(real_images_, real_labels, node_feature=node_feature)
            fake_adv_output = self.discriminator(fake_images_, fake_labels, node_feature=node_feature)

        dis_acml_loss = d_hinge(real_adv_output, fake_adv_output, DDP=None)
        return dis_acml_loss, fake_images_, real_images_, fake_images_aug, real_images_aug

    def train_generator(self, z, fake_labels, node_feature, real_labels, real_images_):
        ## toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.discriminator, grad=False, num_freeze_layers=-1, is_stylegan=False)
        misc.toggle_grad(model=self.generator, grad=True, num_freeze_layers=-1, is_stylegan=False)
        ## Track
        self.generator.apply(misc.track_bn_statistics)

        fake_images_ = self.generator(z=z, label=fake_labels, node_feature=node_feature)
        fake_images_aug = None

        if self.config_file['diffaug'] == True:
            fake_images_aug = diffaug(x=fake_images_, policy='translation,cutout')
            fake_adv_output = self.discriminator(fake_images_aug, fake_labels, node_feature=node_feature)
        else:
            fake_adv_output = self.discriminator(fake_images_, fake_labels, node_feature=node_feature)

        gen_acml_loss = g_hinge(fake_adv_output, DDP=None)
        return gen_acml_loss, fake_images_, fake_images_aug

    @torch.no_grad()
    def make_sample(self, z, fake_labels, node_feature):
        ## toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.generator, grad=False, num_freeze_layers=-1, is_stylegan=False)
        misc.toggle_grad(model=self.discriminator, grad=False, num_freeze_layers=-1, is_stylegan=False)
        ## Untrack
        self.generator.apply(misc.untrack_bn_statistics)
        fake_images_sample = self.generator(z=z, label=fake_labels, node_feature=node_feature)
        return fake_images_sample

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        image, label = batch
        z = torch.randn(label.shape[0], self.generator.z_dim)
        z = z.type_as(image)
        fake_image = self.make_sample(z, label, None)
        self.fid.update(imgs=((image.repeat(1,3,1,1)+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8), real=True)
        self.fid.update(imgs=((fake_image.repeat(1,3,1,1)+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8), real=False)

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        fid = self.fid.compute()
        self.fid.reset()
        self.log("val_fid", fid)
        print(f"GLOBAL STEP : {self.global_step}, FID : {fid}")