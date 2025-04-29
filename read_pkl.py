import pickle
import pandas as pd
from transformers import AutoTokenizer
import pickle

# Open the .pkl file in binary read mode ('rb')
with open("tokenized_train_array.pkl", "rb") as file:
    # Load the data from the .pkl file
    data = pickle.load(file)

    # Now 'data' contains the object that was stored in the .pkl file
    # You can work with 'data' as needed
    print(data[10])
    print(len(data[10]))

    print(data[5])
    print(len(data[5]))


# Load the ESM2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
vocab = tokenizer.get_vocab()

# To print the tokens:
for token, index in vocab.items():
    print(token, index)
