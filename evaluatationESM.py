from transformers import EsmModel, EsmTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def gen_perp_and_embed(csv, device):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)

    df = pd.read_csv(csv)
    perp_all_seq = []
    embedding_all_seq = []
    for _, gen_seq_row in df.iterrows():
        sequence = gen_seq_row['sequence']
        
        inputs = tokenizer(gen_seq_row, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        logits = outputs.logits
        
        #get the hidden size
        print("last hidden shape")
        print(las_hidden_states.shape)
        embedding_all_seq.append(last_hidden_states)

        #remove the last logits distribution as there is nothing after 
        shift_logits = logits[:-1, :].contiguous()
        #get the input id ("true")
        shift_labels = inputs['input_ids'][:, 1:].contiguous()

        #gather shifted logits on the vocab axes to get the corresponding probabililty for the label
        #use view to broadcast dims to be hte same as shift_logits as gather expects this
        torch.gather(shift_logits, 1, shift_labels.view(-1, 1))

        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='mean'
        )

        perplexity = torch.exp(loss).item()
        perp_all_seq.append(perplexity)

    return perp_all_seq, embedding_all_seq


def genTSNE(embed_default_gen, embed_gen, save_dir, plot_name):
    
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

def genBoxPlot(perplexity_data, names, save_dir):
    #perplexity_data is a list and so is name
    plt.figure() 
    plt.boxplot(perplexity_data, labels=names)
    plt.title('Box and Whisker Plot')
    plt.ylabel('Perplexity')

    name = name + '.png'
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)

def main():
    csv_fulldiff_gen = 'fulldif_gen.csv'
    csv_twostepdiff_gen = 'twostepdiff_gen.csv'
    csv_incrementdiff_gen = 'incrementdiff_gen.csv'
    csv_progen_gen = 'progen_gen'
    device = 'cuda'
    save_dir = 'L_data_temp_1_test'

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
    
    genBoxPlot(all_perplexity, all_names, save_dir)

def get_evaluation_set():

    df = pd.read_csv('/home/en540-lludwig2/ProMDLM/data/lysozyme_sequences.csv')

    evaluation_data = df_train[int(n*0.9):]

    df_shuffled_evaluation_data = df_train.sample(frac=1)

    # Select the first 100 rows from the 'sequence' column
    evaluation_seq = df_shuffled_evaluation_data[['Sequence']].head(100)  # double brackets to keep it a DataFrame

    # Save to a new CSV
    evaluation_seq.to_csv('random_100_sequences_test.csv', index=False)


def test_main():
    csv_fulldiff_gen = 'fulldif_gen.csv'
    device = 'cuda'
    save_dir = 'L_data_temp_1_test'

    all_csv = [csv_fulldiff_gen, csv_twostepdiff_gen, csv_incrementdiff_gen, csv_progen_gen]
    all_names = ['full_diff', 'two_step_diff', 'increment', 'progen']

    #start with finding perplexity and embedding of the default
    perplexity_default, embedding_default = gen_perp_and_embed('random_100_sequences_test.csv', device) 

    all_perplexity = [] 
    all_embedding = []

    for csv in all_csv:
        perplexity, embedding = gen_perp_and_embed(csv,device)
        all_perplexity.append(perplexity)
        all_embedding.append(embedding)
    
    #generate T-SNE compared to the default
    for num_e, embedding in enumerate(all_embedding):
        genTSNE(embedding_default, embedding, save_dir, all_names[num_e])
    
    genBoxPlot(all_perplexity, all_names, save_dir)

if __name__ == "__main__":




    




    