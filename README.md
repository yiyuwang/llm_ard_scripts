# LLM Cannabis Use Behavior in Clinical Notes

This repository contains the code and data for the LLM_Cannabis_EHR project. The aim of this project is to describe cannabis use behaviors in EHR

## Quickstart:

# 
1. set up conda env and download models from hf
```shell
# Create env (if not already created)
conda create -n llm_ard python=3.10 -y

# Activate env
conda activate llm_ard

# Install from requirements.txt
pip install -r requirements.txt

# Then upgrade huggingface_hub (optional, if you want latest)
pip install -U huggingface_hub

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
--local-dir ./LLMs/Meta-Llama-3.1-8B-Instruct \
--local-dir-use-symlinks False
```
2. run scripts
```shell
mkdir output/logs

nohup python main_reason_combined_batch.py > output/logs/main_reason_prompting_{MODEL ID}.log 2>&1 &

nohup python main_json_combined_batch.py > output/logs/main_status_prompting_{MODEL ID}.log 2>&1 &

nohup python main_json_combined_batch_set-full.py > output/logs/main_status_batch_{model_id}.log 2>&1 &

nohup python main_reason_combined_batch_set-full.py > output/logs/main_reason_batch_gptoss20b_multistep.log 2>&1 &
```
