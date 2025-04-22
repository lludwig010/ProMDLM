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


def compute_pppl(sequence, model, alphabet, offset_idx):
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())

    return sum(log_probs)

def gen_pseudo_perp_seq(sequence, model, tokenizer):
    
    
    total_log_likelihood = 0.0
    tokenized_input = tokenizer(sequence, return_tensors="pt")
    input_ids = tokenized_input['input_ids'][0]

    # Skip special tokens ([CLS], [SEP]) from being masked
    for i in range(1, len(input_ids) - 1):

        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = tokenizer.mask_token_id
        masked_inputs = {
            "input_ids": masked_input_ids.unsqueeze(0).to(device),
            "attention_mask": tokenized_input['attention_mask'].to(device)
        }

        with torch.no_grad():
            outputs = model(**masked_inputs)
            logits = outputs.logits

        softmax_logits = F.log_softmax(logits[0, i], dim=-1)

        true_token_id = input_ids[i].item()
        log_prob = softmax_logits[true_token_id].item()

        total_log_likelihood += log_prob

    avg_neg_log_likelihood = -total_log_likelihood / (len(input_ids) - 2)

    pseudo_perplexity = np.exp(avg_neg_log_likelihood)

    

    return pseudo_perplexity

def gen_pseudo_perp_csv(model, tokenizer, csv, device, temp_use):
    df = pd.read_csv(csv)
    if 'temperature' in df.columns:
        df = df[df['temperature'] == temp_use]

    print("len df:")
    print(len(df))

    all_perplexity = []
    for row_num, gen_seq_row in df.iterrows():
        #print(row_num)
        sequence = gen_seq_row['sequence']
        pseudo_perplexity = gen_pseudo_perp(sequence, model, tokenizer)
        all_perplexity.append(pseudo_perplexity)
    
    return all_perplexity
    


def gen_embed_csv(model, tokenizer, csv, device,temp_use):
    df = pd.read_csv(csv)

    if 'temperature' in df.columns:
        df = df[df['temperature'] == temp_use]

    print("len df:")
    print(len(df))

    embedding_all_seq = []
    for row_num, gen_seq_row in df.iterrows():
        #print(row_num)
        sequence = gen_seq_row['sequence']

        # Tokenize original input and get embeddings
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            pooled_embedding = last_hidden_states.mean(dim=0)  # [hidden_dim]
            embedding_all_seq.append(pooled_embedding.cpu().numpy())
        
    return embedding_all_seq


def gen_pseudo_perp_and_embed(csv, device):
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)
    model.eval()

    df = pd.read_csv(csv)
    perp_all_seq = []
    embedding_all_seq = []

    print("vocab")
    for token, idx in tokenizer.get_vocab().items():
        print(f"{token}: {idx}")

    for row_num, gen_seq_row in df.iterrows():
        print(row_num)
        sequence = gen_seq_row['sequence']

        # Tokenize original input and get embeddings
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            pooled_embedding = last_hidden_states.mean(dim=0)  # [hidden_dim]
            embedding_all_seq.append(pooled_embedding.cpu().numpy())

        #gen the pseudo perpelxity
        #UNCOMMENT THIS
        pseudo_perplexity = gen_pseudo_perp(sequence, model, tokenizer)

        '''
        # Pseudo-perplexity calculation
        total_log_likelihood = 0.0
        tokenized_input = tokenizer(sequence, return_tensors="pt")
        input_ids = tokenized_input['input_ids'][0]

        
        # Skip special tokens ([CLS], [SEP]) from being masked
        #print("sequence")
        #print(sequence)
        #print(input_ids)
        for i in range(1, len(input_ids) - 1):
            #print("num on in seq")
            #print(i)
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = tokenizer.mask_token_id
            masked_inputs = {
                "input_ids": masked_input_ids.unsqueeze(0).to(device),
                "attention_mask": tokenized_input['attention_mask'].to(device)
            }

            with torch.no_grad():
                outputs = model(**masked_inputs)
                logits = outputs.logits
            #print("logits for the masked")
            #print(logits[0, i])
            softmax_logits = F.log_softmax(logits[0, i], dim=-1)
            #print("log softmax logits")
            #print(softmax_logits)
            true_token_id = input_ids[i].item()
            log_prob = softmax_logits[true_token_id].item()
            #print("log_prob")
            #print(log_prob)
            total_log_likelihood += log_prob

        avg_neg_log_likelihood = -total_log_likelihood / (len(input_ids) - 2)
        #print("avg neg_log_likelihood")
        #print(avg_neg_log_likelihood)
        pseudo_perplexity = np.exp(avg_neg_log_likelihood)
        #print("pseudo perplexity")
        #print(pseudo_perplexity)
        '''
        perp_all_seq.append(pseudo_perplexity)

    return perp_all_seq, embedding_all_seq

def genTSNECorrect(all_embeding, save_dir, plot_name, all_names):
    tsne = TSNE(n_components=2, random_state=42)
    # Combine all embeddings into one array
    combined_embeddings = np.vstack(all_embeding)

    # Generate labels for plotting
    labels = []
    for i, emb in enumerate(all_embeding):
        labels.extend([all_names[i]] * len(emb))

    # Apply t-SNE to all data at once
    transformed = tsne.fit_transform(combined_embeddings)

    # Plot using different colors for each family
    plt.figure(figsize=(10, 8))
    colors = cm.get_cmap('tab10', len(all_embeding))

    for i, name in enumerate(all_names):
        #all indices where the label is the name
        idxs = [j for j, label in enumerate(labels) if label == name]
        plt.scatter(
            transformed[idxs, 0],
            transformed[idxs, 1],
            alpha=0.7,
            color=colors(i),
            label=name
        )

    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.title(f"{plot_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, plot_name + ".png"))
    plt.close()


def genTSNE(all_embeding, save_dir, plot_name, all_names):
    
    tsne = TSNE(n_components=2, random_state=42)
    colors = cm.get_cmap('tab10', len(all_embeding))  # choose colormap and number of colors
    #transformed_vectors_default_gen = tsne.fit_transform(embed_default_gen)
    #transformed_vectors_gen = tsne.fit_transform(embed_gen)

    # Plot results
    plt.figure(figsize=(8, 6))
    #plt.scatter(transformed_vectors_default_gen[:, 0], transformed_vectors_default_gen[:, 1], c='blue', alpha=0.7)
    #plt.scatter(transformed_vectors_gen[:, 0], transformed_vectors_gen[:, 1], c='red', alpha=0.7)
    for num, embeding in enumerate(all_embeding):
        embeding_array = np.array(embeding)
        transformed_vectors_default_gen = tsne.fit_transform(embeding_array)
        plt.scatter(transformed_vectors_default_gen[:, 0], transformed_vectors_default_gen[:, 1], alpha=0.7, label=f"{all_names[num]}")
        

    plt.xlabel("t-SNE_1")
    plt.ylabel("t-SNE_2")
    plt.title(f"{plot_name}")
    plt.legend()

    plot_name = plot_name + '.png'
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path)

def genBoxPlot(perplexity_data, names, box_plot_name, save_dir):
    #perplexity_data is a list and so is name
    plt.figure() 
    plt.boxplot(perplexity_data, labels=names)
    plt.title('Box and Whisker Plot')
    plt.ylabel('Perplexity')


    box_plot_name = box_plot_name + '.png'
    save_path = os.path.join(save_dir, box_plot_name)
    plt.savefig(save_path)

def main():
    device = 'cuda'
    save_dir = 'L_evaluation_incrementDiff_NEW'
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)
    model.eval()
    
    temp_use = 1.5
    
    all_csv_tSNE = ['/home/en540-lludwig2/ProMDLM/generated_sequences/lysozyme_100_test_set_final_results_full_t1.5_filtered.csv', '/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_increment_results_full_t1.5_filtered_results_esmfold.csv','/home/en540-lludwig2/ProMDLM/misc_ProteinData/claudins_sequences.csv','/home/en540-lludwig2/ProMDLM/misc_ProteinData/histones_sequences.csv','/home/en540-lludwig2/ProMDLM/misc_ProteinData/kinases_sequences.csv','/home/en540-lludwig2/ProMDLM/misc_ProteinData/ribosomes_sequences.csv']
    all_names_tSNE = ['lysozymes', 'incrementDiff', 'claudins', 'histones', 'kinases', 'ribosomes']

    all_csv_perp = ['']
    all_names_perp = ['']

    all_perplexity = [] 
    all_embedding = []

    for csv in all_csv_tSNE:
        print(f"doing: {csv}")
        embedding = gen_embed_csv(model, tokenizer, csv, device, temp_use)
        all_embedding.append(embedding)
    '''
    for csv in all_csv_perp:
        pseudo_perpelexity = gen_pseudo_perp_csv(model, tokenizer, csv, device, temp_use)
        all_perplexity.append(pseudo_perpelexity)
    '''

    name_plot = f'tSNE visualization of incrementDiff with temp {temp_use}'
    #genTSNE(all_embedding, save_dir, name_plot, all_names_tSNE)
    genTSNECorrect(all_embedding, save_dir, name_plot, all_names_tSNE)
    
    #box_plot_name = f'box_plot_pseudo_perplexity_t{temp_use}'
    #genBoxPlot(all_perplexity, all_names, box_plot_name, save_dir)



def get_evaluation_set():

    df = pd.read_csv('/home/en540-lludwig2/ProMDLM/data/lysozyme_sequences.csv')

    n = len(df)

    def_eval = df[int(n*0.9):]

    df_shuffled_evaluation_data = def_eval.sample(frac=1)

    # Select the first 100 rows from the 'sequence' column
    evaluation_seq = df_shuffled_evaluation_data[['Sequence']].head(200)  # double brackets to keep it a DataFrame

    # Save to a new CSV
    evaluation_seq.to_csv('lysozyme_500_sequences_test.csv', index=False)

def add_pseudo_perplexity_to_CSV(csv, device):
    name = csv.split('.')[0]

    print('name:')
    print(name)
    
    df = pd.read_csv(csv)
    all_pseudo_perplexity = []

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)

    for _, row in df.iterrows():
        sequence = row['sequence']
        pseudo_perplexity = gen_pseudo_perp_seq(sequence, model, tokenizer)
        all_pseudo_perplexity.append(pseudo_perplexity)
    
    df['pseudo_perplexity'] = all_pseudo_perplexity
    df.to_csv(name + '_w_pseudo_perplexity.csv', index=False)

if __name__ == "__main__":
    main()
    '''
    csv = '/home/en540-lludwig2/ProMDLM/generated_sequences/lysozyme_100_test_set_final.csv'
    device = 'cuda'
    add_pseudo_perplexity_to_CSV(csv, device)
    #get_evaluation_set()

    #tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    #model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    #model = model.to(device)

    
    seq_rand = 'VHMHWVEKKARPAEGEPKVFEEEVILPLKSIHPRFGDQDGTMCEESNAVIIEQFDFBLDICTPASRPACSGSFASVDYTWEIYPMRLRDAWFEMYTQFEFQCPNFGVIIEFARRDKASFYKQCDHWWPYYVWACNWEADCRESFSFTVCKVA'
    seq_true = 'ISSATVNLIKGSESLVPIPSPDPIGLLTVGYGHKCLKPQCSEVTFPFPLSSSTASQLFAQDMTQYINCLHRSISKSVVLNDNQFGALVSWTYNAGCEGMGTSTLVKRLNNGEDPNTVVAQELPKWNIAKKKISKGLVNRRNREISFFQTPSNVVAHPLC'
    perp = gen_pseudo_perp_seq(seq_rand, model, tokenizer)
    print('perp rand')
    print(perp)

    perp = gen_pseudo_perp_seq(seq_true, model, tokenizer)
    print('perp true')
    print(perp)
    '''
    




    




    