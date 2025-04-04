import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from model import LinearModel, LinearModelLeakyReLU
from scheduler import noise_schedule, scheduler_loss_fn, apply_noise

# Get the absolute path to the directory containing the other file
from lactamase.Dataset import CustomDataset



def train_loop(model, train_loader, optimizer, loss_fn, epochs, max_timesteps, batch_size, seq_len, vocab_size, device):

    train_losses, train_accuracy, train_precision, train_recall, train_F1 = [], [], [], [], []
    
    for epoch in range(epochs):
        print(f"doing epoch: {epoch}")

        model.train()
        total_loss = 0.0

        num = 0

        #load in the batches of sequences
        for batched_sequences_tokenized in train_loader:
            optimizer.zero_grad()

            # sample time step
            t = torch.randint(0, max_timesteps, (1,)).item()

            batch_masks = noise_schedule(max_timesteps, t, batch_size, seq_len)
            masked_batch_seq, batch_masks = apply_noise(batch_masks, batched_sequences_tokenized, t)
            batch_pred_tokens = model(masked_batch_seq)
            batch_loss = scheduler_loss_fn(batch_pred_tokens, batched_sequences_tokenized, masked_batch_seq, vocab_size)

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

            num+= 1
    
        avg_train_loss = total_loss / len(train_loader)
        print(f"epoch loss: {avg_train_loss}")
        train_losses.append(avg_train_loss)

        
    return train_losses, train_accuracy, train_precision, train_recall, train_F1, model

# will probably need to add config inputs here
def train_main(device, learning_rate, batch_size, num_epochs):

    #Path to train and test pkl files
    train_file_pkl = 'tokenized_train_array.pkl'
    # Dont actually need a test
    test_file_pkl = 'tokenized_train_array.pkl'

    # Alter for the kind of model we want

    #TODO Ghassan builds the model 
    
    model = PROMDLM() # Changes
    model = model.to(device)
 
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Load Mushroom Dataset
    train_dataset = CustomDataset(train_file_pkl)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses, model = train_loop(model, train_loader, optimizer, loss, num_epochs, device)

    #TODO define plot function
    #plot_results(train_losses)

    return train_losses, model




if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_main()