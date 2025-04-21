import os
import pandas as pd

def csv_to_fasta(csv_file, fasta_file, sequence_col, id_col= None):
    """
    Transforms a CSV file with sequences into a FASTA file.

    Args:
        csv_file (str): Path to the input CSV file.
        fasta_file (str): Path to the output FASTA file.
        sequence_col (str): Column name containing sequences.
        id_col (str): Column name containing unique identifiers.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    if id_col is None:
        # If no id_col is provided, use the index as the identifier
        df[id_col] = df.index

    # Open the FASTA file for writing
    with open(fasta_file, 'w') as fasta:
        for _, row in df.iterrows():
            # Write the FASTA format
            fasta.write(f">{row[id_col]}\n{row[sequence_col]}\n")