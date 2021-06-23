

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl


def load_model(params):
    if params.gen == 'dcgan':
        from dcgan import Generator
        gen = Generator(latent=params.latent, features=params.features)

    if params.disc == 'dcgan':
        from dcgan import Discriminator
        disc = Discriminator(features=params.features)

    return gen, disc


class GAN(pl.LightningModule):
    def __init__(self, b1: float = 0.5, b2: float = 0.999, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.generator, self.discriminator = load_model(self.hparams)
        self.validation_z = torch.randn(min(self.hparams.batch, 144), self.hparams.latent, 1, 1)  # Same vector to make the timelapse

    def forward(self, z):
        return self.generator(z)

    def latent(self, batch):
        z = torch.randn(batch, self.hparams.latent, 1, 1).type_as(self.generator.gen[0][0].weight)
        return z

    def sample(self, num):
        z = self.latent(num)
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.train()
        # Train discriminator
        if optimizer_idx == 0:
            # Real images are labeled 1
            imgs, _ = batch
            yhat = self.discriminator(imgs)
            valid = torch.ones_like(yhat)
            real_loss = self.adversarial_loss(yhat, valid)
            # Fake images are labeled 0
            z = self.latent(self.hparams.batch)
            yhat = self.discriminator(self.generator(z).detach())
            fake = torch.zeros_like(yhat)
            fake_loss = self.adversarial_loss(yhat, fake)
            # Discriminator loss is the average of these
            d_loss = (real_loss+fake_loss)/2
            tqdm_dict = {'d_loss': d_loss}
            self.logger.experiment.add_scalar("d_loss", d_loss, global_step=self.global_step)
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
            })
            return output

        # Train generator
        if optimizer_idx == 1:
            z = self.latent(self.hparams.batch)
            generated_imgs = self.generator(z)
            # Use the real label so the generator gets the right gradients        
            yhat = self.discriminator(generated_imgs)
            valid = torch.ones_like(yhat)
            g_loss = self.adversarial_loss(yhat, valid)
            tqdm_dict = {'g_loss': g_loss}
            self.logger.experiment.add_scalar("g_loss", g_loss, global_step=self.global_step)
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        disc_multi = 0.1
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=disc_multi*lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def on_epoch_end(self):
        self.eval()
        z = self.validation_z.type_as(self.generator.gen[0][0].weight)
        # log sampled images
        sample_imgs = self.generator(z)
        grid = make_grid(sample_imgs, nrow=16)
        grid = (grid+1)/2.0
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
