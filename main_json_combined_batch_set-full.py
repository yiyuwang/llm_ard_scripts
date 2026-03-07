import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import string
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import re, json, math, time
from transformers.generation.logits_process import LogitsProcessor


from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import logging
logging.set_verbosity_info()


print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

from prompts_status import get_prompts


def get_model(model_id):
    # Path to your locally downloaded model
    # model_path = f"LLMs/{model_id}"
    # Load tokenizer and model
    if "gemma" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(f"LLMs/{model_id}",
            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            f"LLMs/{model_id}",
            torch_dtype=torch.float32,
            device_map="auto",
            local_files_only=True,
            attn_implementation="eager"
        )
    elif "gpt-oss" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(f"LLMs/{model_id}",
            local_files_only=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            f"LLMs/{model_id}",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            attn_implementation="eager",
            trust_remote_code=True
        )
   
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"LLMs/{model_id}",
            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            f"LLMs/{model_id}",
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            attn_implementation="eager"
        )
        

    # Create text generation pipeline

    return model, tokenizer

def generate_with_fallback(model, prompt_text, temp):
    """Generate text with fallback options if primary method fails"""
    try:
        # Primary generation approach
        model(
            prompt_text, 
            max_new_tokens=100, 
            temperature=temp,
            do_sample=False
            )
    except RuntimeError as e:
        print(f"Error with primary generation: {e}")
        try:
            print("First fallback: Try with greedy decoding")
            return model(prompt_text, max_new_tokens=100, do_sample=False)
        except RuntimeError as e2:
            print("Second fallback: Try with eager attention implementation")
            try:
                return model(prompt_text,max_new_tokens=100, temperature=temp,  attn_implementation="eager")
            except RuntimeError as e3:
                print(f"Error with second fallback: {e3}")
                return model(prompt_text, max_new_tokens=50)



class SanitizeLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        return scores.clamp(min=-1e9, max=1e9)
  
class MinimalSanitizeLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Replace NaNs/Infs and clamp
        scores = scores.clone()
        scores[scores != scores] = -1e9           # NaN to large negative
        scores[scores == float("inf")] = 1e9
        scores[scores == float("-inf")] = -1e9
        scores = scores.clamp(min=-100, max=100)  # prevent overflow
        return scores
    
import torch
import time

def log_gpu_memory(batch_idx, tag=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{tag} Batch {batch_idx}] alloc={alloc:.2f}GB | reserv={reserv:.2f}GB | peak={peak:.2f}GB")


from torch.utils.data import DataLoader, Dataset
class TextDataset(Dataset):
    def __init__(self, texts, ids, prompt_fn):
        """
        texts: list[str]
        ids: list[str or int] - unique identifiers for each snippet
        prompt_fn: function that takes text -> prompt string
        """
        self.texts = texts
        self.ids = ids
        self.prompt_fn = prompt_fn

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "uid": self.ids[idx],
            "prompt": self.prompt_fn(self.texts[idx]),
            "text": self.texts[idx]
        }



def extract_label_from_response(text):
    """
    Tries to pull the integer classification from the model's response.
    Prefer regex on the key to avoid brittle full-JSON parsing when there is extra text.
    Returns an int or None.
    """
    m = re.search(r'"classification"\s*:\s*([1-6])', text)
    if m:
        return int(m.group(1))
    # very permissive fallback: look for a bare 1-6 on the last lines
    tail = "\n".join(text.strip().splitlines()[-5:])
    m2 = re.search(r'\b([1-6])\b', tail)
    return int(m2.group(1)) if m2 else None

def batched_inference(data, model, tokenizer, prompt_fn, temp, batch_size=4):
    """
    Expects `data` to have a 'text' column (e.g., a pandas DataFrame).
    Returns (y_pred, responses, json_data).
    - y_pred: list[int or None]
    - responses: list[str] (generated text per item)
    - json_data: list[dict] with {"snippet", "predicted_label", "full_response"}
    """
    # Llama models often lack a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(
        texts=data["text"].tolist(),
        ids=data["uid"].tolist(),
        prompt_fn=prompt_fn
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_pred, responses, json_data = [], [], []

    # for tracking overall time and peak memory
    total_start = time.time()
    total_peak_mem = 0.0
    total_time = 0.0

    for i, batch in enumerate(dataloader):
        # batch is a dict of lists (because PyTorch collates dicts automatically)
        batch_ids = batch["uid"]
        batch_prompts = batch["prompt"]
        batch_snippets = batch["text"]

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=temp,
            do_sample=True,
            renormalize_logits=True,
            logits_processor=LogitsProcessorList([SanitizeLogits()]),
        )

        gen_only = gen[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for uid, snippet, response_text in zip(batch_ids, batch_snippets, decoded):
            pred = extract_label_from_response(response_text)
            json_data.append({
                "uid": uid.item() if torch.is_tensor(uid) else uid,
                "snippet": snippet,
                "predicted_label": pred,
                "full_response": response_text
            })


        torch.cuda.synchronize()
        batch_time = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        total_peak_mem = max(total_peak_mem, peak_gb)
        total_time += batch_time
        print(f"[Batch {i+1}] peak GPU {peak_gb:.2f} GB | time {batch_time:.2f}s")

    overall_time = time.time() - total_start
    avg_batch_time = total_time / len(dataloader)
    current_alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    print("\n========== Inference Summary ==========")
    print(f"Total batches: {len(dataloader)}")
    print(f"Total time: {overall_time:.2f} s")
    print(f"Avg batch time: {avg_batch_time:.2f} s")
    print(f"Peak GPU memory (any batch): {total_peak_mem:.2f} GB")
    print(f"Final GPU allocated: {current_alloc:.2f} GB | reserved: {reserved:.2f} GB")
    print("=======================================\n")

    return json_data



def run_full_text(model_id, strat_name="chain_of_thought", temp=0.7):
    """Compare model performance across different temperatures using JSON format"""
    print("Running temperature comparison with JSON output format")
    
    # Load data
    data = pd.read_csv("data/matched_rows_with_context_buffer-100_uid.csv")
    data = data.rename(columns={"Relevant Snippets": "text"})
    print(data.columns)
    # Get prompt strategies
    strategies = get_prompts()

    from transformers import AutoConfig
    local_dir = f"LLMs/{model_id}"
    print(os.path.exists(os.path.join(local_dir, "config.json")))
    cfg = AutoConfig.from_pretrained(f"LLMs/{model_id}", local_files_only=True)
    print(cfg.model_type)
    
    # Load model
    model, tokenizer = get_model(model_id)
    
    # Create output directories
    os.makedirs(f"output/status_batch/{model_id}", exist_ok=True)
    os.makedirs(f"output/status_batch/{model_id}/figures", exist_ok=True)
    os.makedirs(f"output/status_batch/{model_id}/prediction", exist_ok=True)

    
    # For each strategy
    prompt_fn = strategies[strat_name]
    
    print(f"\nEvaluating strategy: {strat_name}")
    print(f"  Temperature: {temp}")
        
    # Generate response with current temperature
    json_data = batched_inference(data, model, tokenizer, prompt_fn, temp, batch_size=4)

    # Save predictions as JSON
    with open(f"output/status_batch/{model_id}/prediction/{strat_name}_temp{temp}_predictions.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    

if __name__ == "__main__":
    # Run the temperature comparison with JSON format
    # run_full_text(model_id="Llama-3.1-8B-Instruct", strat_name="one_shot", temp=0.3)
    # summary = run_full_text(model_id="medgemma-4b-it")
    run_full_text(model_id="gpt-oss-20b", strat_name="chain_of_thought", temp=0.3)
    print("Done!")