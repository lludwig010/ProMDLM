import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PROMDLM.scheduler import noise_schedule, scheduler_loss_fn, apply_noise
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
from PROMDLM.lactamase.Dataset import CustomDataset
import logging
import torch
import os

class Trainer:
    def __init__(self, model, optimizer, loss_function, epochs, train_loader, val_loader, max_timesteps, batch_size, seq_len, vocab_size, device, output_dir, job_name="TEST"):

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


    def train_loop_fullDiff(self):
        # Training scheme where amount diffuse increases with time
        
        train_losses = []
        val_losses = [] 
        train_losses_batch = []
        val_losses_batch=[]
        print("num train data points:")
        print(len(self.train_loader))

        for self.epoch in range(self.epochs):
            print(f"doing epoch: {self.epoch}")

            self.model.train()
            train_total_loss = 0.0
            val_total_loss = 0.0

            num = 0


            #load in the batches of sequences
            self.model.train()
            
            for batched_sequences_tokenized in self.train_loader:
                #print(f"batch num {num}")

                batch_size = batched_sequences_tokenized.shape

                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,1:-1].to(self.device)
                self.optimizer.zero_grad()

                # sample time step
                t = torch.randint(0, self.max_timesteps, (1,)).item()

                #print(f"sampled timestep {t}")

                batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                batch_pred_tokens = self.model(masked_batch_seq)
                batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                batch_loss.backward()
                #print(f"batch loss: {batch_loss}")
                self.optimizer.step()

                train_losses_batch.append(batch_loss.item())

                train_total_loss += batch_loss.item()

                num+= 1

            self.model.eval()
            #print("Validation")
            torch.cuda.empty_cache()
            for batched_sequences_tokenized in self.val_loader:

                batch_size = batched_sequences_tokenized.shape
                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,1:-1].to(self.device)

                # sample time step
                t = torch.randint(0, self.max_timesteps, (1,)).item()

                batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                batch_pred_tokens = self.model(masked_batch_seq)
                val_batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                val_total_loss += val_batch_loss.item() 
                val_losses_batch.append(val_batch_loss.item()) 
        
            avg_train_loss = train_total_loss / len(self.train_loader)
            print(f"epoch loss train: {avg_train_loss}")
            train_losses.append(avg_train_loss)

            avg_val_loss = val_total_loss/len(self.val_loader) 
            print(f"epoch loss val: {avg_train_loss}")
            val_losses.append(avg_val_loss)


        plt.figure()
        plt.plot(train_losses_batch)
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Train Batch Loss")
        plt.savefig(self.output_dir + '/batch_loss_train.png')
  
        plt.figure()
        plt.plot(val_losses_batch)
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Val Batch Loss")
        plt.savefig(self.output_dir + '/batch_loss_val.png')

        plt.figure()
        plt.plot(train_losses)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Train epoch Loss")
        plt.savefig(self.output_dir + '/epoch_loss_train.png')

        plt.figure()
        plt.plot(val_losses)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Val epoch Loss")
        plt.savefig(self.output_dir + '/epoch_loss_val.png')

        #save the weights at the end of training
        torch.save(self.model, self.output_dir + f'/{self.job_name}_weights.pth')

 
        return train_losses, self.model

    def train_increment_diffusion(self):
        # Training scheme where amount diffuse increases with time
        
        train_losses = []
        val_losses = [] 
        train_losses_batch = []
        val_losses_batch=[]
        print("num train data points:")
        print(len(self.train_loader))

        logging.basicConfig(
            filename='training_increment.log',
            filemode='w',  # 'a' for append, 'w' to overwrite each time
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logger = logging.getLogger()

        logger.info("num train data points: %d", len(self.train_loader))

        
        for epoch in range(self.epochs):
            print(f"doing epoch: {epoch}")
            logger.info(f"Starting epoch: {epoch}")

            self.model.train()
            train_total_loss = 0.0
            val_total_loss = 0.0

            num = 0
            #load in the batches of sequences
            self.model.train()
            
            for batched_sequences_tokenized in self.train_loader:
                if (num % 10 == 0):
                    print(f"batch num {num}")
                    logger.info(f"Training batch num: {num}")

                batch_size = batched_sequences_tokenized.shape

                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,1:-1].to(self.device)
                self.optimizer.zero_grad()

                # sample time step based off of the epoch training is on 
                # timestep is proportional to current epoch on
                timestep = (epoch/self.epochs) * self.max_timesteps

                # define range for possible timesteps to sample between. upper and lower bound are 10% away from current timestep
                low_timestep = max(0, timestep - 0.1 * self.max_timesteps)
                max_timestep = min(self.max_timesteps, timestep + 0.1 * self.max_timesteps)

                '''
                print(low_timestep)
                print(type(low_timestep))
                print(max_timestep)
                print(type(max_timestep))
                '''

                t = torch.randint(int(low_timestep), int(max_timestep), (1,)).item()

                #print(f"sampled timestep {t}")

                batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                batch_pred_tokens = self.model(masked_batch_seq)
                batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                batch_loss.backward()
                #print(f"batch loss: {batch_loss}")
                self.optimizer.step()

                train_losses_batch.append(batch_loss.item())

                train_total_loss += batch_loss.item()

                num+= 1

            self.model.eval()
            print("Validation")
            '''
            torch.cuda.empty_cache()
            for batched_sequences_tokenized in self.val_loader:
                if (num % 10 == 0):
                    print(f"batch num {num}")

                batch_size = batched_sequences_tokenized.shape
                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,1:-1].to(self.device)

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
                '''
        
            avg_train_loss = train_total_loss / len(self.train_loader)
            print(f"epoch loss: {avg_train_loss}")
            logger.info(f"Epoch {epoch} average training loss: {avg_train_loss:.4f}")
            train_losses.append(avg_train_loss)

            '''
            avg_val_loss = val_total_loss/len(self.val_loader) 
            print(f"epoch loss val: {avg_train_loss}")
            val_losses.append(avg_val_loss)
            '''

        plot(self.job_name, train_losses_batch, val_losses_batch, train_losses, val_losses, self.output_dir)

        torch.save(self.model, self.output_dir + f'/{self.job_name}_weights.pth')
        logger.info("Training completed and model saved.")
    



def plot(job_name, batch_train_losses, batch_val_losses, epoch_train_loses, epoch_val_losses, output_dir):
    # function to plot the epoch and batch loss graphs, saved under output_dir with the job name

        plt.figure()
        plt.plot(batch_train_losses)
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Train Batch Loss")
        plt.savefig(output_dir + f'/{job_name}_batch_loss_train.png')

        '''
        plt.figure()
        plt.plot(batch_val_losses)
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Val Batch Loss")
        plt.savefig(output_dir + f'/{job_name}_batch_loss_val.png')
        '''

        plt.figure()
        plt.plot(epoch_train_loses)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Train epoch Loss")
        plt.savefig(output_dir + f'/{job_name}_epoch_loss_train.png')

        '''
        plt.figure()
        plt.plot(epoch_val_losses)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Val epoch Loss")
        plt.savefig(output_dir + f'/{job_name}_epoch_loss_val.png')
        '''



def train_main():
    config = OmegaConf.load("configs/L_config.yaml")
    os.makedirs(config.paths.output_dir, exist_ok=True)

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
    # added job name for naming runs
    job_name = config.paths.job_name
    output_dir = config.paths.output_dir



    cfg = OmegaConf.load(config.paths.pretrained_model_cfg)
    model = DiffusionProteinLanguageModel.from_pretrained(
        config.paths.pretrained_model_name, cfg_override=cfg
    ).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = CustomDataset(train_file_pkl, max_datapoints=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_file_pkl)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    trainer = Trainer(
        model, optimizer, loss, num_epochs, train_loader, val_loader,
        max_timesteps, batch_size, seq_len, vocab_size, device, output_dir, job_name
    )


    # default training scheme
    if training_scheme == 1:
        train_losses, model = trainer.train_loop_fullDiff()
    # incremental diffusion
    elif training_scheme == 2:
        train_losses, model = trainer.train_increment_diffusion()

    print("Training complete. Saving model.")
    model.save_pretrained(output_dir)

    return train_losses, model



if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_main()