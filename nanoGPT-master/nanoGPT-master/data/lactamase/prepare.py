import os
import numpy as np
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

'''
df_train = pd.read_csv('Users/lludw/Documents/GrayLab_Class/finalProj/ProMDLM/ESM_traindata.csv')

# add CLS token at the end to all sequences
data_train = df_train['Protein Sequences'] + ['CLS']


n = len(df_train)
train_data = data_train[:int(n*0.9)]
val_data = data_train[int(n*0.9):]
'''
# encode with autotokenizer with custom vocab
# make tokenizer
#enc = tiktoken.get_encoding("gpt2")

vocab = [
    "<mask>", "<unk>",  
    "A", "R", "N", "ASP", "C", "E", "Q", "G", "H", "I", 
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]

token2id = {token: idx for idx, token in enumerate(vocab)}
id2token = {idx: token for token, idx in token2id.items()}

tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))

# Set the special tokens
tokenizer.add_special_tokens(["<mask>","<unk>"])

# Pre-tokenizer defines how sequences are split into tokens
tokenizer.pre_tokenizer = pre_tokenizers.CharLevel()

tokenizer.model = models.WordLevel(token2id, unk_token="<unk>")

sequence = "ARN<mask>HE"

encoded = tokenizer.encode(sequence)


print("Encoded:", encoded.ids, encoded.tokens)


'''
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens'
'''