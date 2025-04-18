from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
import torch


#cfg = OmegaConf.load("configs/config_150m.yaml")
#model = DiffusionProteinLanguageModel.from_pretrained("facebook/esm2_t30_150M_UR50D", cfg_override=cfg, from_huggingface=False)
model = torch.load("/home/jtso3/ghassan/ProMDLM/training_results/big_two_stage_training/big_two_stage_training_weights.pth", weights_only=False)
device = torch.device("cuda:0")
model = model.to(device)
tokenizer =  AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

generation_length = 282
nb_generated_sequences = 4


input_string= "cls " + "L "* generation_length + "eos"
input_id_one_seq = tokenizer.encode(input_string)
input_ids = torch.tensor([input_id_one_seq] * nb_generated_sequences)
input_ids = input_ids.to(device)
batch = {
    "input_ids": input_ids,
}
output = model.generate(batch, max_iter=100, temperature=1, resample_ratio=0.5, sampling_strategy = "vanilla")

for i in range(nb_generated_sequences):
    print("sequence", i)
    decode = tokenizer.decode(output[0][i], skip_special_tokens=True)
    print(decode.replace(" ",""))




