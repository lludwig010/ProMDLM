import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from models.scheduler import noise_schedule, scheduler_loss_fn, apply_noise
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
from models.Dataset import CustomDataset
import logging
import torch
import os
import time
import shutil

class Trainer:
    def __init__(self, model, optimizer, loss_function, epochs, train_loader, val_loader, max_timesteps, batch_size, seq_len, vocab_size, device, output_dir, job_name="TEST", validation = False):

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_timesteps = max_timesteps
        self. batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.output_dir = output_dir
        self.job_name = job_name
        self.validation = validation
        self.logger = logging.getLogger()


    def train_loop_fullDiff(self):
        # Training scheme where amount diffuse increases with time
        
        train_losses = []
        val_losses = [] 
        train_losses_batch = []
        val_losses_batch=[]

        for self.epoch in range(self.epochs):

            self.logger.info(f"doing epoch: {self.epoch} out of {self.epochs}")

            self.model.train()
            train_total_loss = 0.0
            val_total_loss = 0.0
            nb_batches_train = len(self.train_loader)

            #load in the batches of sequences
            self.model.train()
            
            for i, batched_sequences_tokenized in enumerate(self.train_loader):
                if i%10 ==0: self.logger.info(f"Training batch: {i}/{nb_batches_train}")

                batch_size = batched_sequences_tokenized.shape
                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,:].to(self.device)  # remove eos?
                self.optimizer.zero_grad()

                # sample time step
                t = torch.randint(0, self.max_timesteps, (1,)).item()

                if i%10 ==0: self.logger.info(f"sampled timestep {t}")

                batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                batch_pred_tokens = self.model(masked_batch_seq)
                batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                batch_loss.backward()

                try:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True) # gradient clipping
                    self.optimizer.step()
                    if i%10 ==0: self.logger.info(f"batch loss: {batch_loss}")
                    train_losses_batch.append(batch_loss.item())
                    train_total_loss += batch_loss.item()


                except RuntimeError as e:
                    self.logger.warning(f"GRADIENTS EXPLODED FOR BATCH {i} TIMESTEP {t}, WE DO NOT UPDATE THE WEIGHTS.")
                    percent_masked = torch.sum(batch_masks == 1).item() / batch_masks.numel()
                    self.logger.warning(f"percent masked: {percent_masked}")

            
            avg_train_loss = train_total_loss / len(self.train_loader)
            self.logger.info(f"epoch loss train: {avg_train_loss}")
            train_losses.append(avg_train_loss)

            if self.validation:
                self.model.eval()
                self.logger.info("Validation")
                torch.cuda.empty_cache()
                nb_batches_val = len(self.val_loader)
                
                for i, batched_sequences_tokenized in enumerate(self.val_loader):
                    self.logger.info(f"validation batch: {i}/{nb_batches_val}")

                    batch_size = batched_sequences_tokenized.shape
                    size_to_mask = batch_size[0]

                    #remove start and end tokens so that length is 286
                    batched_sequences_tokenized = batched_sequences_tokenized[:,:].to(self.device)  # remove eos?
                    self.optimizer.zero_grad()

                    # sample time step
                    t = torch.randint(0, self.max_timesteps, (1,)).item()

                    batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                    masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                    batch_pred_tokens = self.model(masked_batch_seq)
                    val_batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                    val_total_loss += val_batch_loss.item() 
                    val_losses_batch.append(val_batch_loss.item()) 

                avg_val_loss = val_total_loss/len(self.val_loader) 
                self.logger.info(f"epoch loss val: {avg_train_loss}")
                val_losses.append(avg_val_loss)

        plot(self.job_name, train_losses_batch, val_losses_batch, train_losses, val_losses, self.output_dir)

        #save the weights at the end of training
        torch.save(self.model, self.output_dir + f'/{self.job_name}_weights.pth')

        return train_losses, self.model
    
    def train_loop_two_stage(self):
            # Training scheme where amount diffuse increases with time
            
            train_losses = []
            val_losses = [] 
            train_losses_batch = []
            val_losses_batch=[]

            for self.epoch in range(self.epochs):

                self.logger.info(f"doing epoch: {self.epoch} out of {self.epochs}")

                self.model.train()
                train_total_loss = 0.0
                val_total_loss = 0.0
                nb_batches_train = len(self.train_loader)

                #load in the batches of sequences
                self.model.train()
                
                for i, batched_sequences_tokenized in enumerate(self.train_loader):
                    if i%10==0:
                        self.logger.info(f"Training batch: {i}/{nb_batches_train}")

                    batch_size = batched_sequences_tokenized.shape
                    size_to_mask = batch_size[0]

                    #remove start and end tokens so that length is 286
                    batched_sequences_tokenized = batched_sequences_tokenized[:,:].to(self.device)  # remove eos?
                    self.optimizer.zero_grad()

                    # sample time step
                    if self.epoch < self.epochs/2:
                        # first half of training, 15% noise
                        t = 0.15 * self.max_timesteps
                    else:
                        # second half of training, random noise sampling
                        t = torch.randint(0, self.max_timesteps, (1,)).item()

                    if i%10 == 0: self.logger.info(f"sampled timestep {t}")

                    batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                    masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                    batch_pred_tokens = self.model(masked_batch_seq)
                    batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                    batch_loss.backward()
                    
                    try:
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True) # gradient clipping
                        self.optimizer.step()
                        if i%10 ==0: self.logger.info(f"batch loss: {batch_loss}")
                        train_losses_batch.append(batch_loss.item())
                        train_total_loss += batch_loss.item()

                    except RuntimeError as e:
                        self.logger.warning(f"GRADIENTS EXPLODED FOR BATCH {i} TIMESTEP {t}, WE DO NOT UPDATE THE WEIGHTS.")
                
                avg_train_loss = train_total_loss / len(self.train_loader)
                self.logger.info(f"epoch loss train: {avg_train_loss}")
                train_losses.append(avg_train_loss)

                if self.validation:
                    self.model.eval()
                    self.logger.info("Validation")
                    torch.cuda.empty_cache()
                    nb_batches_val = len(self.val_loader)
                
                    for i, batched_sequences_tokenized in enumerate(self.val_loader):
                        self.logger.info(f"validation batch: {i}/{nb_batches_val}")

                        batch_size = batched_sequences_tokenized.shape
                        size_to_mask = batch_size[0]

                        #remove start and end tokens so that length is 286
                        batched_sequences_tokenized = batched_sequences_tokenized[:,:].to(self.device)  # remove eos?
                        self.optimizer.zero_grad()

                        # sample time step
                        if self.epoch < self.epochs/2:
                            # first half of training, fixed timestep
                            t = 0.15 * self.max_timesteps
                        else:
                            t = torch.randint(0, self.max_timesteps, (1,)).item()

                        batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                        masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                        batch_pred_tokens = self.model(masked_batch_seq)
                        val_batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                        val_total_loss += val_batch_loss.item() 
                        val_losses_batch.append(val_batch_loss.item()) 

                    avg_val_loss = val_total_loss/len(self.val_loader) 
                    self.logger.info(f"epoch loss val: {avg_train_loss}")
                    val_losses.append(avg_val_loss)

            plot(self.job_name, train_losses_batch, val_losses_batch, train_losses, val_losses, self.output_dir)

            #save the weights at the end of training
            torch.save(self.model, self.output_dir + f'/{self.job_name}_weights.pth')

            return train_losses, self.model

    def train_increment_diffusion(self):
        # Training scheme where amount diffuse increases with time
        
        train_losses = []
        val_losses = [] 
        train_losses_batch = []
        val_losses_batch=[]


        self.logger.info("num train data points: %d", len(self.train_loader))

        
        for epoch in range(self.epochs):

            self.logger.info(f"doing epoch: {epoch} out of {self.epochs}")

            self.model.train()
            train_total_loss = 0.0
            val_total_loss = 0.0
            nb_batch_train = len(self.train_loader)

            self.model.train()
            
            for i, batched_sequences_tokenized in enumerate(self.train_loader):
                
                if i%10 ==0: self.logger.info(f"Training batch {i}/{nb_batch_train}")

                batch_size = batched_sequences_tokenized.shape

                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,:].to(self.device)  # remove eos?
                self.optimizer.zero_grad()

                # sample time step based off of the epoch training is on 
                # timestep is proportional to current epoch on
                timestep = ((epoch/self.epochs) + (i/nb_batch_train)*(1/self.epochs))* self.max_timesteps

                # define range for possible timesteps to sample between. upper and lower bound are 10% away from current timestep
                low_timestep = max(0, timestep - 0.1 * self.max_timesteps)
                max_timestep = min(self.max_timesteps, timestep + 0.1 * self.max_timesteps)

                t = torch.randint(int(low_timestep), int(max_timestep), (1,)).item()

                if i%10 ==0: self.logger.info(f"sampled timestep {t}")

                batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                batch_pred_tokens = self.model(masked_batch_seq)
                batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                batch_loss.backward()

                try:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True) # gradient clipping
                    self.optimizer.step()
                    if i%10 ==0: self.logger.info(f"batch loss: {batch_loss}")
                    train_losses_batch.append(batch_loss.item())
                    train_total_loss += batch_loss.item()

                except RuntimeError as e:
                    self.logger.warning(f"GRADIENTS EXPLODED FOR BATCH {i} TIMESTEP {t}, WE DO NOT UPDATE THE WEIGHTS.")

            
            avg_train_loss = train_total_loss / len(self.train_loader)
            self.logger.info(f"epoch loss train: {avg_train_loss}")
            train_losses.append(avg_train_loss)

            if self.validation:

                self.model.eval()
                self.logger.info("Validation")
                torch.cuda.empty_cache()
                nb_batches_val = len(self.val_loader)

                torch.cuda.empty_cache()
                for i, batched_sequences_tokenized in enumerate(self.val_loader):
                    self.logger.info(f"validation batch: {i}/{nb_batches_val}")

                    batch_size = batched_sequences_tokenized.shape
                    size_to_mask = batch_size[0]

                    #remove start and end tokens so that length is 286
                    batched_sequences_tokenized = batched_sequences_tokenized[:,:].to(self.device)  # remove eos?

                    # sample time step

                    timestep = (epoch/self.epochs) * self.max_timesteps

                    # define range for possible timesteps to sample between. upper and lower bound are 10% away from current timestep
                    low_timestep = max(0, timestep - 0.1 * self.max_timesteps)
                    max_timestep = min(self.max_timesteps, timestep + 0.1 * self.max_timesteps)

                    t = torch.randint(int(low_timestep), int(max_timestep), (1,)).item()
                    

                    batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                    masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                    batch_pred_tokens = self.model(masked_batch_seq)
                    val_batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                    val_total_loss += val_batch_loss.item() 
                    val_losses_batch.append(val_batch_loss.item()) 
            
                avg_val_loss = val_total_loss/len(self.val_loader) 
                self.logger.info(f"epoch loss val: {avg_train_loss}")
                val_losses.append(avg_val_loss)
    

        plot(self.job_name, train_losses_batch, val_losses_batch, train_losses, val_losses, self.output_dir)

        torch.save(self.model, self.output_dir + f'/{self.job_name}_weights.pth')
        self.logger.info("Training completed and model saved.")
        return train_losses, self.model
    



def plot(job_name, batch_train_losses, batch_val_losses, epoch_train_loses, epoch_val_losses, output_dir):
    # function to plot the epoch and batch loss graphs, saved under output_dir with the job name

        plt.figure()
        plt.plot(batch_train_losses)
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Train Batch Loss")
        plt.savefig(output_dir + f'/{job_name}_batch_loss_train.png')

        if len(batch_val_losses) > 0:
            plt.figure()
            plt.plot(batch_val_losses)
            plt.xlabel("Batches")
            plt.ylabel("Loss")
            plt.title("Val Batch Loss")
            plt.savefig(output_dir + f'/{job_name}_batch_loss_val.png')
        

        plt.figure()
        plt.plot(epoch_train_loses)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Train epoch Loss")
        plt.savefig(output_dir + f'/{job_name}_epoch_loss_train.png')

        if len(epoch_val_losses) > 0:
            plt.figure()
            plt.plot(epoch_val_losses)
            plt.xlabel("epoch")
            plt.ylabel("Loss")
            plt.title("Val epoch Loss")
            plt.savefig(output_dir + f'/{job_name}_epoch_loss_val.png')
        



def train_main():

    start = time.perf_counter()

    config_path = "configs/training_incremental.yaml"
    config = OmegaConf.load(config_path)
    os.makedirs(config.paths.output_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(config.paths.output_dir, "config_used.yaml")) #copy config file to output dir



    train_file_pkl = config.paths.train_file
    val_file_pkl = config.paths.val_file

    device = config.training.device
    learning_rate = config.training.learning_rate
    batch_size = config.training.batch_size
    num_epochs = config.training.epochs
    max_timesteps = config.training.max_timesteps
    seq_len = config.training.seq_len
    vocab_size = config.training.vocab_size
    weight_decay = config.training.weight_decay
    # added scheme to choose what kind of training
    training_scheme = config.training.scheme
    output_dir = config.paths.output_dir
    validation = config.training.validation
    # no more need to specify both job name and output dir
    job_name = os.path.basename(output_dir.rstrip('/'))

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(output_dir,f'{job_name}.log'),
        filemode='w',  # 'a' for append, 'w' to overwrite each time
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()



    cfg = OmegaConf.load(config.paths.pretrained_model_cfg)
    model = DiffusionProteinLanguageModel.from_pretrained(
        config.paths.pretrained_model_name, cfg_override=cfg
    ).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = CustomDataset(train_file_pkl, max_datapoints= None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_file_pkl, max_datapoints= 10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    trainer = Trainer(
        model, optimizer, loss, num_epochs, train_loader, val_loader,
        max_timesteps, batch_size, seq_len, vocab_size, device, output_dir, job_name, validation
    )


    # default training scheme
    if training_scheme == 1:
        logger.info("Training scheme 1: full diffusion")
        train_losses, model = trainer.train_loop_fullDiff()
    # incremental diffusion
    elif training_scheme == 2:
        logger.info("Training scheme 2: incremental diffusion")
        train_losses, model = trainer.train_increment_diffusion()
    elif training_scheme == 3:
        logger.info("Training scheme 3: two stage training")
        train_losses, model = trainer.train_loop_two_stage()

    logger.info(f"Training complete. Saving model under {output_dir}")
    logger.info(f"Training took {time.perf_counter() - start:.2f} seconds")
    return train_losses, model



if __name__ == "__main__":

    torch.cuda.empty_cache()
    train_main()