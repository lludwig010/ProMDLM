import transformers
from transformers import EsmForProteinFolding, EsmTokenizer, AutoTokenizer
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import os
import numpy as np
import pandas as pd


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def get_plddt_from_pdb(pdb_string):
    lines = pdb_string.split("\n")
    plddt = []
    for line in lines:
        if line.startswith("ATOM"):
            plddt_value = float(line[60:66].strip())
            plddt.append(plddt_value)
    return np.mean(np.array(plddt))


def calculate_esmfold_scores(input_csv):
    df = pd.read_csv(input_csv)
    output_dir = os.path.dirname(input_csv)
    file_name_without_extension = os.path.splitext(os.path.basename(input_csv))[0]
    df["plddt"] = 0
    df["plddt"] = df["plddt"].astype(float)
    df["ptm"] = 0
    df["ptm"] = df["ptm"].astype(float)

    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    device = "cuda"
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    for i, row in df.iterrows():
        sequence = row["sequence"]
        # Tokenize the sequence
        tokenized_input = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        tokenized_input = tokenized_input.to(device)

        # Perform folding and get confidence scores
        with torch.no_grad():
            outputs = model(tokenized_input)
            print(outputs.keys())
            ptm = outputs["ptm"]
            plddt = get_plddt_from_pdb(convert_outputs_to_pdb(outputs)[0])
            df.at[i, "plddt"] = float(plddt)
            df.at[i, "ptm"] = float(ptm.item())

    df.to_csv(
        os.path.join(output_dir, file_name_without_extension + "_results_esmfold.csv"),
        index=False,
    )
    print(
        f"ESMFold analysis completed. Results saved to {os.path.join(output_dir, file_name_without_extension + '_results_esmfold.csv')}"
    )


if __name__ == "__main__":
    input_csv = [
        "generated_sequences/lysozyme_100_test_set_final_results_full_t1.5_filtered.csv",
        "generated_sequences/generated_sequences_progen_results_full_t1.5_filtered.csv",
        "generated_sequences/generated_sequences_two_stage_results_full_t1.5_filtered.csv",
        "generated_sequences/generated_sequences_increment_results_full_t1.5_filtered.csv",
        "generated_sequences/generated_sequences_fulldiff_results_full_t1.5_filtered.csv",
    ]

    # Perform folding and get confidence scores
    for input in input_csv:
        print("processing:", input)
        calculate_esmfold_scores(input)
