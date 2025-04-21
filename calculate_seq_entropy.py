import os 
import pandas as pd
import numpy as np
from utils import csv_to_fasta


def calculate_sequence_entropy(input_csv):

  

    output_dir = os.path.dirname(input_csv)
    file_name_without_extension = os.path.splitext(os.path.basename(input_csv))[0]

    df = pd.read_csv(input_csv)
    df["entropy"] = 0
    for i, row in df.iterrows():
        sequence = row["sequence"]
        amino_acids = list(sequence)
        unique_amino_acids, counts = np.unique(amino_acids, return_counts=True)
        print(unique_amino_acids)
        print(counts)
        probabilities = counts / len(amino_acids)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        df.at[i, "entropy"] = entropy


    df.to_csv(os.path.join(output_dir, file_name_without_extension+ "_results_entropy.csv"), index=False)
    print(f"BLASTP analysis completed. Results saved to {os.path.join(output_dir, file_name_without_extension+ "_results_entropy.csv")}")

if __name__ == "__main__":
    input_csv = ["/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_fulldiff.csv",
                 "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_increment.csv",
                 "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_progen.csv",
                 "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_two_stage.csv",
                 "/home/jtso3/ghassan/ProMDLM/generated_sequences/lysozyme_100_sequences_test.csv"]

    for input in input_csv:
        calculate_sequence_entropy(input)
