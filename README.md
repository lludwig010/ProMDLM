# Discrete Space Diffusion: A Different Method for Protein Sequence Generation using LLMs

## Table of Contents

- [Overview](#overview)
- [Inference](#Inference)
- [ProGen_Finetune](#ProGen_Finetune)
- [License](#license)

## Overview

This project encompasses our work in applying discrete space diffusion for protein language models to generate developable lysozymes. Building off of the original DPLM paper for the model architecture, we build 3 training schemes to train the model on the sequence diffusion task. The three schemes are Full Diffusion, Two Stage Diffusion and Incremental Diffusion. The first two training schemes are from our implementations of the training schemes detailed in DPLM. Incremental Diffusion is a training scheme we propose to fix some of the limitations in the other two. We find that the model trained on Incremental Diffusion improves its ability to generate outside of its training set while maintaining high foldability and low generative perplexity. 

## Inference

First install the requirements file
```sh
pip install -r requirements.txt
```

Next, download the model weights from each training scheme [here](https://drive.google.com/drive/folders/1nfjkp3n-Xve_MR8dC0CIKF95sGU8pXd3?usp=drive_link)

Finally, run the following script 'generate_script_commands.py' with the commands lines below:

```bash
python /home/en540-lludwig2/ProMDLM/generate_script_commands.py \
  --weight_path 'path_to_full_diff_weights path_to_increment_diff_weights' \
  --save_names 'full_diff increment_diff'  \
  --temp '0.5 1 1.5' \
  --save_dir 'my_outputs_new' \
  --max_iter 100 \
  --generation_length 160 \
  --nb_generated_sequences 1 \
  --resample_ratio 0.3 \
  --device 'cuda'
```

## ProGen_Finetune

In our evaluation we finetune ProGen to compare to our sequence space diffusion approach. Code for finetuning ProGen can 
be found [here](https://github.com/hugohrban/ProGen2-finetuning).

## License
MIT License. See LICENSE file for details.

