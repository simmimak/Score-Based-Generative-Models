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

from pytorch_lightning.loggers import WandbLogger

import wandb

hyperparameter_defaults = dict(
    batch_size = 32,
    learning_rate = 1e-3,
    optimizer = 'adam',
    epochs = 1
)


wandb.init(config=hyperparameter_defaults)

config = wandb.config


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

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs =[]

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
        if config.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr= config.learning_rate)
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
        self.log('train_loss', loss)

        self.training_step_outputs.append({'train_loss': loss})# , 'bpd': bpd.sum() , 'bpd_items': bpd.shape[0]})

        return {'loss': loss, 'log': tensorboard_logs}


    def on_train_epoch_end(self):
      avg_loss         = torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean()   
      #all_bpd =torch.stack([x['bpd'] for x in self.training_step_outputs]).sum()

      #all_items = torch.stack([torch.tensor(x['bpd_items']) for x in self.training_step_outputs]).sum()
      self.log('avg_train_loss', avg_loss)
      #wandb.log('avg_bpd', all_bpd/all_items)
      
      return {'avg_train_loss': avg_loss}
    

    def train_dataloader(self):

        train_loader = DataLoader(self.train_set,
                                batch_size=self.conf.training.dataloader.batch_size,
                                shuffle=True,
                                num_workers=self.conf.training.dataloader.num_workers,
                                pin_memory=True,
                                drop_last=self.conf.training.dataloader.drop_last)
        
        print(len(train_loader))

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
        _, bpd = ode_likelihood(x, self.model, x.shape[0], eps)
    
        #all_bpds += bpd.sum()
        #all_items += bpd.shape[0]

        #avg_bpd = bpd.sum() / bpd.shape[0]

        self.log('val_loss', loss)

        self.validation_step_outputs.append({'val_loss': loss , 'bpd': bpd.sum() , 'bpd_items': bpd.shape[0]})
    
        return {'val_loss': loss}
    

    def on_validation_epoch_end(self):
        avg_loss         = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        all_bpd =torch.stack([x['bpd'] for x in self.validation_step_outputs]).sum()

        all_items = torch.stack([torch.tensor(x['bpd_items']) for x in self.validation_step_outputs]).sum()

        #tensorboard_logs = {'val_loss': avg_loss , 'avg_bpd': all_bpd/all_items}

        sample_batch_size = 64

        samples = ode_sampler(self.model, sample_batch_size)

        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        wandb.log({'val_images': [wandb.Image(sample_grid, caption='MNIST')]})
        wandb.log({'avg_bpd': all_bpd/all_items})


        print('val_avg_bpd', all_bpd/all_items)

        self.validation_step_outputs = []

        return {'val_loss': avg_loss ,'avg_bpd': all_bpd/all_items}
    

    def test_step(self, batch, batch_nb):

        sample_batch_size = 64
        samples = ode_sampler(self.model,
                              sample_batch_size)

        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        #self.logger.experiment.add_image(f'generated_images', sample_grid, self.current_epoch)


        x = (x * 255. + torch.rand_like(x)) / 256.    
        _, bpd = ode_likelihood(x, self.model,
                                x.shape[0], eps=1e-5)

        self.log('testbpd', bpd)
    
        self.test_step_outputs.append({'bpd': bpd.sum() , 'bpd_items': bpd.shape[0]})

        return {'bpd': bpd.sum() , 'bpd_items': bpd.shape[0]}


    def on_test_epoch_end(self):

      all_bpd =torch.stack([x['bpd'] for x in self.test_step_outputs]).sum()
      all_items = torch.stack([torch.tensor(x['bpd_items']) for x in self.test_step_outputs]).sum()
      tensorboard_logs = {'test_avg_bpd': all_bpd/all_items}
      self.log('test_avg_bpd', all_bpd/all_items)

      return {'log': tensorboard_logs}
 

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

    wandb_logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("/content/sample_data/ckptdir"),
                                           filename="best",
                                            monitor='avg_bpd',
                                            mode = 'min', 
                                            verbose=False,
                                            save_last=True,
                                            save_top_k=1,
                                            save_weights_only=True,
                                            
                                            )
    
    trainer = pl.Trainer(fast_dev_run= False,logger=wandb_logger, accelerator ="gpu",devices=args.n_gpu, strategy="ddp", max_epochs= config.epochs,num_nodes= 1,check_val_every_n_epoch=10 ,callbacks=checkpoint_callback)

    trainer.fit(score_based_diffusion_model)








    






    


    
