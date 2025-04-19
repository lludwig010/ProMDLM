from transformers import EsmModel, EsmTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def genTSNE(embed_default_gen, embed_gen, plot_name):
    
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
    plt.savefig(f"tSNE_plot_{plot_name}")

def genDensityPerp(perplexity, name):
    sns.kdeplot(data=perplexity, x=f'Perplexity_{name}', fill=True)
    plt.title('Shaded Density Plot of Value')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f"perp_density_plot_{name}")

if __name__ == "__main__":
    gen_default_csv = 'default.csv'
    gen_new_csv = 'gen.csv'
    device = 'cude'
    perp_default, embed_default = gen_perp_and_embed(gen_default_csv,device)
    perp_new, embed_new = gen_perp_and_embed(gen_new_csv,device)
    
    genTSNE()



    