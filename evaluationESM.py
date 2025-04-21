from transformers import EsmForMaskedLM, EsmTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import os 


def compute_pppl(sequence, model, alphabet, offset_idx):
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())

    return sum(log_probs)

def gen_pseudo_perp(sequence, model, tokenizer):
    
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
    

    return pseudo_perplexity


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
        sequence = gen_seq_row['Sequences']

        # Tokenize original input and get embeddings
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            pooled_embedding = last_hidden_states.mean(dim=0)  # [hidden_dim]
            embedding_all_seq.append(pooled_embedding.cpu().numpy())

        #gen the pseudo perpelxity
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

def gen_perp_and_embed(csv, device):
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)

    df = pd.read_csv(csv)
    perp_all_seq = []
    embedding_all_seq = []
    for _, gen_seq_row in df.iterrows():
        sequence = gen_seq_row['Sequences']
        
        inputs = tokenizer(sequence, return_tensors="pt")
        
        inputs = inputs.to(device)

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        print("sequence length")
        print(len(sequence))
        print("logit shapes")
        print(logits.shape)
        logits = logits[0]
        #get the hidden size
        print("last hidden shape")
        print(last_hidden_states.shape)
        single_last_hidden_states = last_hidden_states[0]
        pooled_last_hideen_state = torch.mean(single_last_hidden_states, dim=0)
        print("pooled last hidden shape")
        print(pooled_last_hideen_state.shape)
        embedding_all_seq.append(pooled_last_hideen_state.detach().cpu().numpy())

        #remove the last logits distribution as there is nothing after 
        shift_logits = logits[:-1, :].contiguous()
        #get the input id ("true")

        print('input_ids shape')
        print(inputs['input_ids'].shape)
        print(inputs['input_ids'][0].shape)
        shift_labels = inputs['input_ids'][0][1:].contiguous()

        #gather shifted logits on the vocab axes to get the corresponding probabililty for the label
        #use view to broadcast dims to be hte same as shift_logits as gather expects this
        torch.gather(shift_logits, 1, shift_labels.view(-1, 1))
        
        print("shifted logits shape")
        print(shift_logits.shape)

        loss = F.cross_entropy(
            shift_logits,
            reduction='mean'
        )

        perplexity = torch.exp(loss).item()
        perp_all_seq.append(perplexity)

    return perp_all_seq, embedding_all_seq


def genTSNE(embed_default_gen, embed_gen, save_dir, plot_name):

    #fit to np.array first
    embed_default_gen = np.array(embed_default_gen)
    embed_gen = np.array(embed_gen)
    
    tsne = TSNE(n_components=2, random_state=42)
    transformed_vectors_default_gen = tsne.fit_transform(embed_default_gen)
    transformed_vectors_gen = tsne.fit_transform(embed_gen)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_vectors_default_gen[:, 0], transformed_vectors_default_gen[:, 1], c='blue', alpha=0.7)
    plt.scatter(transformed_vectors_gen[:, 0], transformed_vectors_gen[:, 1], c='red', alpha=0.7)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE Visualization {plot_name}")

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
    csv_fulldiff_gen = 'fulldif_gen.csv'
    csv_twostepdiff_gen = 'twostepdiff_gen.csv'
    csv_incrementdiff_gen = 'incrementdiff_gen.csv'
    csv_progen_gen = 'progen_gen'
    device = 'cuda'
    save_dir = 'L_data_temp_1_test'

    os.makedirs(save_dir, exist_ok=True)

    all_csv = [csv_fulldiff_gen, csv_twostepdiff_gen, csv_incrementdiff_gen, csv_progen_gen]
    all_names = ['full_diff', 'two_step_diff', 'increment', 'progen']

    #start with finding perplexity and embedding of the default
    perplexity_default, embedding_default = gen_perp_and_embed('default csv', device) 

    all_perplexity = [] 
    all_embedding = []

    for csv in all_csv:
        perplexity, embedding = gen_perp_and_embed(csv,device)
        all_perplexity.append(perplexity)
        all_embedding.append(embedding)
    
    #generate T-SNE compared to the default
    for num_e, embedding in enumerate(all_embedding):
        genTSNE(embedding_default, embedding, save_dir, all_names[num_e])
    
    box_plot_name = 'box_plot_perplexity'
    genBoxPlot(all_perplexity, box_plot_name, save_dir)

def get_evaluation_set():

    df = pd.read_csv('/home/en540-lludwig2/ProMDLM/data/lysozyme_sequences.csv')

    n = len(df)

    def_eval = df[int(n*0.9):]

    df_shuffled_evaluation_data = def_eval.sample(frac=1)

    # Select the first 100 rows from the 'sequence' column
    evaluation_seq = df_shuffled_evaluation_data[['Sequences']].head(100)  # double brackets to keep it a DataFrame

    # Save to a new CSV
    evaluation_seq.to_csv('lysozyme_100_sequences_test.csv', index=False)

def add_pseudo_perplexity_to_CSV(csv):
    name = csv.split('.')[0]
    
    df = pd.read_csv(csv)
    all_pseudo_perplexity = []

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)

    for _, row in df.itereows():
        sequence = gen_seq_row['Sequences']
        pseudo_perplexity = gen_pseudo_perp(sequence, model, tokenizer)
        all_pseudo_perplexity.append(pseudo_perplexity)
    
    df['pseudo_perplexity'] = all_pseudo_perplexity
    evaluation_seq.to_csv(name + '_w_pseudo_perplexity.csv', index=False)


def test_main():
    csv_fulldiff_gen = '/home/en540-lludwig2/ProMDLM/bulk_gen/fulldiff_temp1/fulldiff_gen_t1.csv'
    device = 'cuda'
    save_dir = 'L_data_temp_1_test'

    all_csv = ['/home/en540-lludwig2/ProMDLM/bulk_gen/fulldiff_temp0.5/fulldiff_gen_t0.5.csv', csv_fulldiff_gen, '/home/en540-lludwig2/ProMDLM/bulk_gen/fulldiff_temp1.5/fulldiff_gen_t1.5.csv', '/home/en540-lludwig2/ProMDLM/bulk_gen/fulldiff_temp2/fulldiff_gen_t2.csv', '/home/en540-lludwig2/ProMDLM/lysozyme_100_sequences_test.csv']
    all_names = ['full_diff_t0.5', 'full_diff_t1', 'full_diff_t1.5', 'full_diff_t2', 'test_set']

    os.makedirs(save_dir, exist_ok=True)

    #start with finding perplexity and embedding of the default
    perplexity_default, embedding_default = gen_pseudo_perp_and_embed('lysozyme_100_sequences_test.csv', device) 

    all_perplexity = [] 
    all_embedding = []

    for csv in all_csv:
        perplexity, embedding = gen_pseudo_perp_and_embed(csv,device)
        all_perplexity.append(perplexity)
        all_embedding.append(embedding)
    
    #generate T-SNE compared to the default
    for num_e, embedding in enumerate(all_embedding):
        genTSNE(embedding_default, embedding, save_dir, all_names[num_e])
    
    box_plot_name = 'box_plot_pseudo_perplexity'
    genBoxPlot(all_perplexity, all_names, box_plot_name, save_dir)

if __name__ == "__main__":
    test_main()




    




    