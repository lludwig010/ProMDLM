from transformers import AutoTokenizer
from omegaconf import OmegaConf
from models import DiffusionProteinLanguageModel
import torch
import pandas as pd
import os
import argparse
import logging


parser = argparse.ArgumentParser(description='A simple program to demonstrate argparse')
parser.add_argument('--weight_path', nargs='+', type=str, help='List of weight paths to different models you wish to use', default=[
    '/home/en540-lludwig2/ProMDLM/training_results/L_scheme_1_full_lysozyme/L_scheme_1_full_lysozyme_weights.pth'])
parser.add_argument('--save_names', nargs='+', type=str, default=["full_diff"], help='Names For each weight')  
parser.add_argument('--temp', nargs='+', type=float, default=[0.5, 1, 1.5], help='List of temperatures to use')
parser.add_argument('--save_dir', type=str, default='my_outputs', help='Save directory to use for outputs')
parser.add_argument('--max_iter', type=int, default=500, help='Number of iterations to used for denoising')
parser.add_argument('--generation_length', type=int, default=150, help='Number of tokens in generation')
parser.add_argument('--nb_generated_sequences', type=int, default=1, help='Number of sequences to generate')
parser.add_argument('--resample_ratio', type=float, default=0.2, help='Ratio of common tokens to resample generation')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')

args = parser.parse_args()


weight_path = args.weight_path 
temp_list = args.temp
save_dir = args.save_dir
save_names = args.save_names
output_dir = args.save_dir
max_iter = args.max_iter
generation_length = args.generation_length
nb_generated_sequences = args.nb_generated_sequences
resample_ratio = args.resample_ratio
device = args.device

os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "generation_log.txt")

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

logger.info("Starting generation script")

for model_path, save_name in zip(weight_path, save_names):
    dict_list = []

    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    for temperature in temp_list:
        logger.info(f"Generating sequences with temperature {temperature} for model {save_name}")

        input_string = "cls " + "L " * generation_length + "eos"
        input_id_one_seq = tokenizer.encode(input_string)
        input_ids = torch.tensor([input_id_one_seq] * nb_generated_sequences).to(device)

        batch = {"input_ids": input_ids}
        output = model.generate(
            batch,
            max_iter=max_iter,
            temperature=temperature,
            resample_ratio=resample_ratio,
            sampling_strategy="vanilla",
        )

        for i in range(nb_generated_sequences):
            decode = tokenizer.decode(output[0][i], skip_special_tokens=True).replace(" ", "")
            dict_list.append({
                "model": save_name,
                "temperature": temperature,
                "sequence": decode
            })

    df = pd.DataFrame(dict_list)
    csv_path = f"{output_dir}/generated_sequences_{save_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved sequences to {csv_path}")

logger.info("Generation completed.")