import os 
import pandas as pd
import numpy as np
from utils import csv_to_fasta

def create_database(db_name, db_path):
    """
    Create a BLAST database from a FASTA file.
    """
    makeblastdb_command = f"makeblastdb -in {db_path} -dbtype prot -out {db_name}"
    os.system(makeblastdb_command)

def run_blastp_analysis(input_csv, db_name):

    df = pd.read_csv(input_csv)

    output_dir = os.path.dirname(input_csv)
    file_name_without_extension = os.path.splitext(os.path.basename(input_csv))[0]
    fasta_filename = os.path.join(output_dir, "temp_seq_file.fasta")
    csv_to_fasta(input_csv, fasta_filename, sequence_col="sequence") #the fasta id is the index of the csv file

    blastp_output = os.path.join(output_dir, "blastp_output.txt")
    blastp_command = f"./blast/bin/blastp -query {fasta_filename} -db {db_name} -out {blastp_output} -outfmt '6 qseqid sseqid pident evalue' -max_target_seqs 1 -max_hsps 1"
    os.system(blastp_command)

    with open(blastp_output, "r") as f:
        lines = f.readlines()

    blast_results = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            blast_results.append({
                "qseqid": int(parts[0]),
                "sseqid": parts[1],
                "pident": float(parts[2]),
                "evalue": float(parts[3])
            })
    blast_df = pd.DataFrame(blast_results).set_index("qseqid")
    merged_df = df.join(blast_df, how="left")

    merged_df.fillna({ "pident": 0, "evalue": 10, "sseqid": "-"}, inplace=True)

    merged_df.to_csv(os.path.join(output_dir, file_name_without_extension+ "_results_blastn.csv"), index=False)
    os.remove(fasta_filename)
    os.remove(blastp_output)
    print(f"BLASTP analysis completed. Results saved to {os.path.join(output_dir, input_csv)}")

if __name__ == "__main__":
    input_csv = ["generated_sequences/generated_sequences_fulldiff.csv",
                 "generated_sequences/generated_sequences_increment.csv",
                 "generated_sequences/generated_sequences_progen.csv",
                 "generated_sequences/generated_sequences_two_stage.csv",
                 "enerated_sequences/lysozyme_100_sequences_test.csv"]

    db_name = "./blast/bin/original_train_set"
    for input in input_csv:
        run_blastp_analysis(input, db_name)




