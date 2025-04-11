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
import os

class Trainer:
    def __init__(self, model, optimizer, loss_function, epochs, train_loader, val_loader, max_timesteps, batch_size, seq_len, vocab_size, device, output_dir):

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

    def train_loop_fullDiff(self):
        
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
                print(f"batch num {num}")

                batch_size = batched_sequences_tokenized.shape

                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,1:-1].to(self.device)
                self.optimizer.zero_grad()

                # sample time step
                t = torch.randint(0, self.max_timesteps, (1,)).item()

                print(f"sampled timestep {t}")

                batch_masks = noise_schedule(self.max_timesteps, t, size_to_mask, self.seq_len)
                masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
                batch_pred_tokens = self.model(masked_batch_seq)
                batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, self.vocab_size)

                batch_loss.backward()
                print(f"batch loss: {batch_loss}")
                self.optimizer.step()

                train_losses_batch.append(batch_loss.item())

                train_total_loss += batch_loss.item()

                num+= 1

            self.model.eval()
            print("Validation")
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
            print(f"epoch loss: {avg_train_loss}")
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

 
        return train_losses, self.model


def train_main():
    config = OmegaConf.load("configs/g_config.yaml")
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
    output_dir = config.paths.output_dir


    cfg = OmegaConf.load(config.paths.pretrained_model_cfg)
    model = DiffusionProteinLanguageModel.from_pretrained(
        config.paths.pretrained_model_name, cfg_override=cfg
    ).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = CustomDataset(train_file_pkl, max_datapoints=100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_file_pkl)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    trainer = Trainer(
        model, optimizer, loss, num_epochs, train_loader, val_loader,
        max_timesteps, batch_size, seq_len, vocab_size, device, output_dir
    )

    train_losses, model = trainer.train_loop_fullDiff()
    return train_losses, model



if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_main()