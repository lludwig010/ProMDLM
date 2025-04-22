from transformers import EsmForMaskedLM, EsmTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os 

def findHits(csv_list, name_list, temp, save_dir):

    all_hit_num = []
    for csv in csv_list:
        df = pd.read_csv(csv)
        #if 'temperature' in df.columns:
        df = df[df['temperature'] == temp]

        print(len(df))

        print('len pident')
        #print(len(df['pident'] <= ))
        
        #hit_df = df.loc[(df['pident'] < 0.8)]
        hit_df = df[(df['pident'] <= 80) & (df['plddt'] >= 0.7)]

        print(len(hit_df))

        num_hits = len(hit_df)
        all_hit_num.append(num_hits)
    

    plt.bar(name_list, all_hit_num, color = 'skyblue', width = 0.5) 
    plt.xlabel("Trained Model")
    plt.ylabel("Num Hits")
    plt.title(f"Num Hits Acoss Different Training Schemes At Temp {temp}")
    plt.xticks(name_list)
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    plt.savefig(os.path.join(save_dir, f"hits_temp{temp}.png"))
    plt.close()

    
if __name__ == "__main__":
    save_dir = "L_hits_eval"
    os.makedirs(save_dir, exist_ok=True)
    csv_list = ['/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_fulldiff_results_full_t1.5_filtered_results_esmfold.csv', '/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_two_stage_results_full_t1.5_filtered_results_esmfold.csv', '/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_increment_results_full_t1.5_filtered_results_esmfold.csv', '/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_progen_results_full_t1.5_filtered_results_esmfold.csv']
    names = ['fullDiff', 'two_stage', 'increment', 'progen']
    temp = 1.5
    findHits(csv_list, names, temp, save_dir)