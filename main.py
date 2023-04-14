import os
import json
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from model import UNet
from diffusion import VariancePreservingSDE, ode_sampler , ode_likelihood
import dataset as dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

class ScoreBasedDiffusion(pl.LightningModule):

    def __init__(self, conf):
        super().__init__()

        self.conf  = conf
        self.save_hyperparameters()

        self.model = UNet(input_channels= self.conf.model.in_channel,
                          input_height=self.conf.model.height,
                          ch= self.conf.model.ch,
                          ch_mult=self.conf.model.channel_multiplier,
                          num_res_blocks=self.conf.model.n_res_blocks,
                          attn_resolutions= (16,),
                          resamp_with_conv=True,
                          dropout=self.conf.model.dropout,
                          )

        T1 = torch.nn.Parameter(torch.FloatTensor([self.conf.model.sde.T]), requires_grad=False)
        
        self.sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T1)

    def forward(self , x, t):
        score = self.model(x , t)
        return score

    def setup(self, stage):
        self.train_set, self.valid_set , self.test_set = dataset.get_train_data(self.conf)


    def configure_optimizers(self):
        if self.conf.training.optimizer.type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.conf.training.optimizer.lr)
        else:
            raise NotImplementedError

        return optimizer


    def training_step(self, batch, batch_nb):

        eps = 1e-5

        x , _ = batch

        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = self.sde.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}
    

    def train_dataloader(self):

        train_loader = DataLoader(self.train_set,
                                batch_size=self.conf.training.dataloader.batch_size,
                                shuffle=True,
                                num_workers=self.conf.training.dataloader.num_workers,
                                pin_memory=True,
                                drop_last=self.conf.training.dataloader.drop_last)

        return train_loader
    
    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.conf.validation.dataloader.batch_size,
                                  shuffle=False,
                                  num_workers=self.conf.validation.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.conf.validation.dataloader.drop_last)

        return valid_loader
    
    def test_dataloader(self):

        test_loader = DataLoader(self.test_set,
                                batch_size=self.conf.test.dataloader.batch_size,
                                shuffle=False,
                                num_workers=self.conf.test.dataloader.num_workers,
                                pin_memory=True,
                                drop_last=self.conf.test.dataloader.drop_last)
        
        return test_loader

    def validation_step(self, batch, batch_nb):

        eps = 1e-5

        x , _ = batch

        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = self.sde.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))

        x = (x * 255. + torch.rand_like(x)) / 256.    
        _, bpd = ode_likelihood(x, self.model, self.sde.marginal_prob_std,
                                self.sde.g, x.shape[0], eps)
    
        all_bpds += bpd.sum()
        all_items += bpd.shape[0]

        avg_bpd = all_bpds / all_items
    
        return {'val_loss': loss , 'avg_bpd': avg_bpd}
    

    def validation_epoch_end(self, outputs):
        avg_loss         = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        sample_batch_size = 64

        samples = ode_sampler(self.model, sample_batch_size)

        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        self.logger.experiment.add_image(f'generated_images', sample_grid, self.current_epoch)
        
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    

    def test_step(self, batch, batch_nb):

        sample_batch_size = 64
        samples = ode_sampler(self.model,
                              sample_batch_size)

        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        self.logger.experiment.add_image(f'generated_images', sample_grid, self.current_epoch)


        x = (x * 255. + torch.rand_like(x)) / 256.    
        _, bpd = ode_likelihood(x, self.model, self.sde.marginal_prob_std,
                                self.sde.g,
                                x.shape[0], eps=1e-5)
    
        all_bpds += bpd.sum()
        all_items += bpd.shape[0]

        avg_bpd = all_bpds / all_items

        tensorboard_logs = {'avg_bpd': avg_bpd}

        return {'avg_bpd': avg_bpd ,'log': tensorboard_logs}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False, help="Training or evaluation?")
    parser.add_argument("--config", type=str, required=True, help="Path to config.")

    # Training specific args
    parser.add_argument("--ckpt_dir", type=str, default='ckpts', help="Path to folder to save checkpoints.")
    parser.add_argument("--ckpt_freq", type=int, default=20, help="Frequency of saving the model (in epoch).")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of available GPUs.")

    args = parser.parse_args()

    path_to_config = args.config
    with open(path_to_config, 'r') as f:
        conf = json.load(f)

    conf = obj(conf)

    score_based_diffusion_model = ScoreBasedDiffusion(conf)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.ckpt_dir, 'sde_{epoch:02d}-{val_loss:.2f}'),
                                            monitor='val_loss',
                                            verbose=False,
                                            save_last=True,
                                            save_top_k=-1,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=args.ckpt_freq,
                                            prefix='')
    
    trainer = pl.Trainer(fast_dev_run=False,
                            gpus=args.n_gpu,
                            max_steps=conf.training.n_iter,
                            precision=conf.model.precision,
                            gradient_clip_val=1.,
                            progress_bar_refresh_rate=20,
                            checkpoint_callback=checkpoint_callback)

    trainer.fit(score_based_diffusion_model)

    trainer.test()




    

















    






    


    