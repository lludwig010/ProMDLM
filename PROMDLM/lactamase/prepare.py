import os
import re
import numpy as np
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizer import AminoAcidTokenizer
import pickle


def tokenize_df(sequence):
    tokens = tokenizer.tokenize(sequence)
    return tokens
    

df_train = pd.read_csv('Users/lludw/Documents/GrayLab_Class/finalProj/ProMDLM/ESM_traindata.csv')

# Add X padding
max_length = df_train['Protein Sequences'].str.len().max()

for index, row in df_train.iterrows():
    #sequence 
    seq = row['Protein Sequences']

    pad_to_add = (max_length - len(seq)) * 'Z'

    df_train.loc[index, 'Protein Seqeunces'] = seq + pad_to_add

n = len(df_train)
print("num data points")
print(n)
train_data = df_train[:int(n*0.9)]
val_data = df_train[int(n*0.9):]


# Z is padding, X is masked
#sequence = "MKTLLLTLVVVTIVCLDLGYTXZ"
tokenizer = AminoAcidTokenizer()

# tokenize train and val sequences 
tokenized_train = pd.DataFrame({'Protein Sequence Tokenized': train_data['Protein Sequences'].tokenize_df(train_data)})
tokenized_val = pd.DataFrame({'Protein Sequence Tokenized': val_data['Protein Sequences'].tokenize_df(val_data)})

tokenized_train_array = tokenized_train['Protein Sequence Tokenized'].to_numpy()
tokenized_val_array = tokenized_val['Protein Sequence Tokenized'].to_numpy()

with open('tokenized_train_array.pkl', 'wb') as f:
    pickle.dump(tokenized_train_array, f)

with open('tokenized_val_array.pkl', 'wb') as f:
    pickle.dump(tokenized_val_array, f)

