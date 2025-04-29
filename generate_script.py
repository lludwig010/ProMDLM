from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
import torch
import pandas as pd
import os


model_paths = [
    "training_results/second_attempt_big_increment_training/second_attempt_big_increment_training_weights.pth",
    "training_results/second_attempt_big_two_stage_training/second_attempt_big_two_stage_training_weights.pth",
    "/home/jtso3/ghassan/ProMDLM/training_results/fulldiff_weights.pth",
]
save_names = ["increment", "two_stage", "fulldiff"]
output_dir = "generated_sequences_215aa_long"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda")

# PARAMETERS
max_iter = 500
generation_length = 150
nb_generated_sequences = 10
resample_ratio = 0.2

for model_path, save_name in zip(model_paths, save_names):

    dict_list = []
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    for temperature in [0.5, 1, 1.5]:
        print(
            f"Generating sequences with temperature {temperature} for model {save_name}"
        )
        input_string = "cls " + "L " * generation_length + "eos"
        input_id_one_seq = tokenizer.encode(input_string)
        input_ids = torch.tensor([input_id_one_seq] * nb_generated_sequences)
        input_ids = input_ids.to(device)
        batch = {
            "input_ids": input_ids,
        }
        output = model.generate(
            batch,
            max_iter=max_iter,
            temperature=temperature,
            resample_ratio=resample_ratio,
            sampling_strategy="vanilla",
        )

        for i in range(nb_generated_sequences):
            decode = tokenizer.decode(output[0][i], skip_special_tokens=True)
            decode = decode.replace(" ", "")
            dict_list.append(
                {"model": save_name, "temperature": temperature, "sequence": decode}
            )

    df = pd.DataFrame(dict_list)
    df.to_csv(f"{output_dir}/generated_sequences_{save_name}.csv", index=False)
