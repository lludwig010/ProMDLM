import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PROMDLM.scheduler import noise_schedule, scheduler_loss_fn, apply_noise
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
from PROMDLM.lactamase.Dataset import CustomDataset

class Trainer:
    def __init__(self, model, optimizer, loss_function, epochs, train_loader, max_timesteps, batch_size, seq_len, vocab_size, device):

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.train_loader = train_loader
        self.max_timesteps = max_timesteps
        self. batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device

    def train_loop_fullDiff(self):
        
        train_losses = []
        for self.epoch in range(self.epochs):
            print(f"doing epoch: {self.epoch}")

            self.model.train()
            total_loss = 0.0

            num = 0

            #load in the batches of sequences
            for batched_sequences_tokenized in self.train_loader:
                print("batche seq")
                print(batched_sequences_tokenized)

                batch_size = batched_sequences_tokenized.shape

                size_to_mask = batch_size[0]

                #remove start and end tokens so that length is 286
                batched_sequences_tokenized = batched_sequences_tokenized[:,1:-1].to(self.device)
                print("batched seq shape:")
                print(batched_sequences_tokenized.shape)
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

                total_loss += batch_loss.item()

                num+= 1
        
            avg_train_loss = total_loss / len(self.train_loader)
            print(f"epoch loss: {avg_train_loss}")
            train_losses.append(avg_train_loss)

            
        return train_losses, self.model

# will probably need to add config inputs here
def train_main():

    #Path to train and test pkl files
    train_file_pkl = '/mnt/c/Users/lludw/Documents/GrayLab_Class/finalProj/ProMDLM/PROMDLM/lactamase/tokenized_train_array.pkl'

    device = 'cuda'
    learning_rate = 0.0001
    batch_size = 10
    num_epochs = 10
    max_timesteps = 100
    seq_len = 286
    vocab_size = 33

    # Alter for the kind of model we want
    cfg = OmegaConf.load("configs/config_150m.yaml")
    model = DiffusionProteinLanguageModel.from_pretrained("facebook/esm2_t30_150M_UR50D", cfg_override=cfg)
    model = model.to(device)
 
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Load Mushroom Dataset
    train_dataset = CustomDataset(train_file_pkl)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    trainer = Trainer(model, optimizer, loss, num_epochs, train_loader, max_timesteps, batch_size, seq_len, vocab_size, device)

    train_losses, model = trainer.train_loop_fullDiff()
    

    #TODO define plot function
    #plot_results(train_losses)

    return train_losses, model




if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_main()