from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
import os
import shutil
import torch
import pandas as pd


df = pd.DataFrame(columns=["Sequences"])

config_path = "configs/gen_fulldiff_bulk.yaml"
config = OmegaConf.load(config_path)
os.makedirs(config.paths.output_dir, exist_ok=True)
shutil.copy(
    config_path, os.path.join(config.paths.output_dir, "config_used.yaml")
)  # copy config file to output dir


# cfg = OmegaConf.load("configs/config_150m.yaml")
# model = DiffusionProteinLanguageModel.from_pretrained("facebook/esm2_t30_150M_UR50D", cfg_override=cfg, from_huggingface=False)
model = torch.load(config.model.path_weights, weights_only=False)
device = torch.device(config.model.device)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

generation_length = config.model.length
nb_generated_sequences = config.model.num_gen

input_string = "cls " + "L " * generation_length + "eos"
input_id_one_seq = tokenizer.encode(input_string)
input_ids = torch.tensor([input_id_one_seq] * nb_generated_sequences)
input_ids = input_ids.to(device)
batch = {
    "input_ids": input_ids,
}
output = model.generate(
    batch,
    max_iter=100,
    temperature=conffig.model.temp,
    resample_ratio=0.5,
    sampling_strategy="vanilla",
)

for i in range(nb_generated_sequences):
    print("sequence", i)
    decode = tokenizer.decode(output[0][i], skip_special_tokens=True)
    print(decode.replace(" ", ""))

    df = pd.concat(
        [df, pd.DataFrame([{"Sequences": decode.replace(" ", "")}])], ignore_index=True
    )

df.to_csv(
    os.path.join(config.paths.output_dir, config.paths.csv_save_name), index=False
)
