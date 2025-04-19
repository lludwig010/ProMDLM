import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import pickle

# Load the ESM2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

# Read the dataset
df_train = pd.read_csv('/home/en540-lludwig2/ProMDLM/data/lysozyme_sequences.csv')

# Find the maximum sequence length to use for padding
max_length = df_train['Sequence'].str.len().max() + 2 # +2 for CLS and EOS tokens

print(f"Maximum sequence length: {max_length}")

# Function to tokenize sequences with proper ESM2 format
def tokenize_sequence(sequence, max_len=None):
    # Format "cls " + "L "* generation_length + "eos"
    # For ESM2, we don't need to manually add these tokens as the tokenizer will handle it

    # Tokenize and convert to input IDs
    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", max_length=max_len)
    return inputs["input_ids"].squeeze().tolist()  # Convert tensor to list for storage

# Split data into train and validation sets
n = len(df_train)
print(f"Total number of data points: {n}")

#shuffle train
shuffled_df_train = df_train.sample(frac=1)

shuffled_df_train.to_csv('data/shuffled_df_train.csv', index=False)

train_data = df_train[:int(n*0.9)]
val_data = df_train[int(n*0.9):]


print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# Tokenize all sequences
tokenized_train = []
for seq in train_data['Sequence']:
    tokenized_train.append(tokenize_sequence(seq, max_length))

tokenized_val = []
for seq in val_data['Sequence']:
    tokenized_val.append(tokenize_sequence(seq, max_length))

# Convert to numpy arrays
tokenized_train_array = np.array(tokenized_train)
tokenized_val_array = np.array(tokenized_val)

print(f"Tokenized train array shape: {tokenized_train_array.shape}")
print(f"Tokenized validation array shape: {tokenized_val_array.shape}")

# Save tokenized data
with open('data/lyzozyme_train_shuffled.pkl', 'wb') as f:
    pickle.dump(tokenized_train_array, f)

with open('data/lyzozyme_val_shuffled.pkl', 'wb') as f:
    pickle.dump(tokenized_val_array, f)

print("Tokenized arrays saved to disk.")