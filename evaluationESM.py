from transformers import EsmForMaskedLM, EsmTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import logging
import os


def gen_pseudo_perp_seq(sequence, model, tokenizer):
    """
    Base function that computes pseudo perplexity from a sequence using a Masked LM
    """

    total_log_likelihood = 0.0
    #  tokenize the input
    tokenized_input = tokenizer(sequence, return_tensors="pt")
    input_ids = tokenized_input["input_ids"][0]

    # skip special tokens ([CLS], [SEP]) from being masked
    for i in range(1, len(input_ids) - 1):
        masked_input_ids = input_ids.clone()
        # set the current token to mask to mask token
        masked_input_ids[i] = tokenizer.mask_token_id
        masked_inputs = {
            "input_ids": masked_input_ids.unsqueeze(0).to(device),
            "attention_mask": tokenized_input["attention_mask"].to(device),
        }

        # Using the masked LM compute the logits
        with torch.no_grad():
            outputs = model(**masked_inputs)
            logits = outputs.logits

        # apply softmax to logits and retrieve only the logit that pertains to the current otken to predict from mask
        softmax_logits = F.log_softmax(logits[0, i], dim=-1)

        # find the correct token id as it is only the probability for this token that we want
        true_token_id = input_ids[i].item()

        # get prob model gave for the actual token
        log_prob = softmax_logits[true_token_id].item()

        total_log_likelihood += log_prob

    avg_neg_log_likelihood = -total_log_likelihood / (len(input_ids) - 2)
    pseudo_perplexity = np.exp(avg_neg_log_likelihood)

    return pseudo_perplexity


def gen_pseudo_perp_csv(model, tokenizer, csv, device, temp_use):
    """
    Function that computes pseudo perplexity from a csv of sequences
    """

    df = pd.read_csv(csv)
    # if the csv has multiple temps, use only the one that has temp desired
    if "temperature" in df.columns:
        df = df[df["temperature"] == temp_use]

    all_perplexity = []
    for row_num, gen_seq_row in df.iterrows():
        sequence = gen_seq_row["sequence"]
        # find pseudo perplexity for each sequence in the csv
        pseudo_perplexity = gen_pseudo_perp(sequence, model, tokenizer)
        all_perplexity.append(pseudo_perplexity)

    return all_perplexity


def gen_embed_csv(model, tokenizer, csv, device, temp_use):
    """
    Fucntion for generating the embeddings of sequences within a csv
    """
    df = pd.read_csv(csv)

    if "temperature" in df.columns:
        df = df[df["temperature"] == temp_use]

    embedding_all_seq = []
    for row_num, gen_seq_row in df.iterrows():
        sequence = gen_seq_row["sequence"]

        # tokenize original input and get embeddings
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # get [seq_len, hidden_dim] from only looking at the last layer hidden states
            last_hidden_states = outputs.hidden_states[-1][0]
            # get [hidden_dim] from pooling all of the hidden states along the sequence length
            pooled_embedding = last_hidden_states.mean(dim=0)
            embedding_all_seq.append(pooled_embedding.cpu().numpy())

    return embedding_all_seq


def genTSNECorrect(all_embeding, save_dir, plot_name, all_names):
    """
    Given a list of a list of embeddings with a corresponding list of names for each list of embeddings generate a tSNE plot of
    the embedding space
    """

    tsne = TSNE(n_components=2, random_state=42)
    # combine all embeddings into one array
    combined_embeddings = np.vstack(all_embeding)

    labels = []
    for i, emb in enumerate(all_embeding):
        labels.extend([all_names[i]] * len(emb))

    transformed = tsne.fit_transform(combined_embeddings)

    # plot using different colors for each family of embeddings
    plt.figure(figsize=(10, 8))
    colors = cm.get_cmap("tab10", len(all_embeding))

    for i, name in enumerate(all_names):
        # all indices where the label is the name
        idxs = [j for j, label in enumerate(labels) if label == name]
        plt.scatter(
            transformed[idxs, 0],
            transformed[idxs, 1],
            alpha=0.7,
            color=colors(i),
            label=name,
        )

    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.title(f"{plot_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, plot_name + ".png"))
    plt.close()


def genBoxPlot(perplexity_data, names, box_plot_name, save_dir):
    """
    Given a list of a list of perplexity data from different categories, plot them on a box plot
    """

    plt.figure()
    plt.boxplot(perplexity_data, labels=names)
    plt.title("Box and Whisker Plot")
    plt.ylabel("Perplexity")

    box_plot_name = box_plot_name + ".png"
    save_path = os.path.join(save_dir, box_plot_name)
    plt.savefig(save_path)


def main():
    """
    Main function to call the evaluations that are needed
    """

    device = "cuda"
    save_dir = "L_evaluation_incrementDiff_NEW_large"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)
    model.eval()

    temp_use = 1.5

    all_csv_tSNE = [
        "/home/en540-lludwig2/ProMDLM/generated_sequences/lysozyme_100_test_set_final_results_full_t1.5_filtered.csv",
        "/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_increment_results_full_t1.5_filtered_results_esmfold.csv",
        "/home/en540-lludwig2/ProMDLM/misc_ProteinData/claudins_sequences.csv",
        "/home/en540-lludwig2/ProMDLM/misc_ProteinData/histones_sequences.csv",
        "/home/en540-lludwig2/ProMDLM/misc_ProteinData/kinases_sequences.csv",
        "/home/en540-lludwig2/ProMDLM/misc_ProteinData/ribosomes_sequences.csv",
    ]
    all_names_tSNE = [
        "lysozymes",
        "incrementDiff",
        "claudins",
        "histones",
        "kinases",
        "ribosomes",
    ]

    all_csv_tSNE = [
        "path_to_increment",
        "path_to_two_stage",
        "path_to_fulldiff",
        "path_to_progen",
        "path_to_test",
    ]
    all_names_box_plot = ["incrementDiff", "two_stage", "fulldiff", "progen", "test"]

    # plot a tSNE
    for csv in all_csv_tSNE:
        print(f"doing: {csv}")
        embedding = gen_embed_csv(model, tokenizer, csv, device, temp_use)
        all_embedding.append(embedding)

    # plot a box plot
    for csv in all_csv_perp:
        pseudo_perpelexity = gen_pseudo_perp_csv(
            model, tokenizer, csv, device, temp_use
        )
        all_perplexity.append(pseudo_perpelexity)

    name_plot = f"tSNE visualization of incrementDiff with temp {temp_use}"
    genTSNECorrect(all_embedding, save_dir, name_plot, all_names_tSNE)

    box_plot_name = f"box_plot_pseudo_perplexity_t{temp_use}"
    genBoxPlot(all_perplexity, all_names_box_plot, box_plot_name, save_dir)


def get_evaluation_set():
    "from the initial training set (assume not shuffled yet), create a test set"
    df = pd.read_csv("/home/en540-lludwig2/ProMDLM/data/lysozyme_sequences.csv")
    n = len(df)
    def_eval = df[int(n * 0.9) :]
    df_shuffled_evaluation_data = def_eval.sample(frac=1)

    # Select the first 100 rows from the 'sequence' column
    evaluation_seq = df_shuffled_evaluation_data[["Sequence"]].head(
        200
    )  # double brackets to keep it a DataFrame
    # Save to a new CSV
    evaluation_seq.to_csv("lysozyme_500_sequences_test.csv", index=False)


def add_pseudo_perplexity_to_CSV(csv, device):
    """
    Given a csv with sequences, add a new column for the pseudo perplexity of each sequence
    """

    logging.basicConfig(
        filename="pseudo_perplexity_log.txt",
        filemode="a",  # Append mode
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    name = csv.split(".cs")[0]

    logger.info(f"Processing file: {csv}")
    logger.info(f"Output prefix: {name}")

    df = pd.read_csv(csv)
    all_pseudo_perplexity = []

    logger.info("Loading tokenizer and model...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)
    logger.info("Model loaded successfully.")

    for num, row in df.iterrows():
        if num == 80:
            break
        sequence = row["sequence"]
        logger.info(f"Processing sequence index: {num}")
        try:
            pseudo_perplexity = gen_pseudo_perp_seq(sequence, model, tokenizer)
            all_pseudo_perplexity.append(pseudo_perplexity)
        except Exception as e:
            logger.error(f"Error processing sequence at index {num}: {e}")
            all_pseudo_perplexity.append(None)

    df["pseudo_perplexity_LARGE"] = all_pseudo_perplexity
    output_path = name + "_w_pseudo_perplexity_LARGEV2.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # call main for general tSNE and BoxPlot Creation
    # main()

    # call to get an evaluation set from an unshuffled train set
    # get_evaluation_set()

    # adding pseudo perplexity to a csv
    csv = "/home/en540-lludwig2/ProMDLM/generated_sequences/generated_sequences_increment_results_full_t1.5_filtered_results_esmfold.csv"
    device = "cuda"
    add_pseudo_perplexity_to_CSV(csv, device)
