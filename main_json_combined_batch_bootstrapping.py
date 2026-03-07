import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import string
import glob
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
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=500,
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



def run_one_iteration_fn(sampled, model, tokenizer, prompt_fn, temp, batch_size):
    pred_json = batched_inference(sampled, model, tokenizer, prompt_fn, temp, batch_size=batch_size)

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(pred_json)
    if "uid" not in pred_df.columns or "predicted_label" not in pred_df.columns:
        raise ValueError("batched_inference must return JSON objects with 'uid' and 'predicted_label' fields.")

    # Merge predictions with ground truth
    merged = sampled.merge(pred_df[["uid", "predicted_label"]], on="uid", how="left")
    y_true = merged["label"].astype(int).tolist()
    y_pred = merged["predicted_label"].astype(int).tolist()

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")

    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "recall": recall,
        "precision": precision,
    }
    return metrics

def _safe_json_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic rename on POSIX

def _load_completed_iters(out_dir):
    done = set()
    for fp in glob.glob(os.path.join(out_dir, "iter_*.json")):
        name = os.path.basename(fp)
        # iter_005.json
        idx = int(name.split("_")[1].split(".")[0])
        done.add(idx)
    return done

def ci95(x):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def bootstrap_evaluation(
    data: pd.DataFrame,
    model,
    tokenizer,
    prompt_fn,
    temp: float,
    sample_per_class: int = 6,
    n_iterations: int = 100,
    batch_size: int = 1,
    out_dir: str = "temp_bootstrap"
) -> dict:
    """
    Perform bootstrap sampling on the dataset to estimate confidence intervals for classification metrics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least 'uid', 'text', and 'label' columns.
    model : transformers.PreTrainedModel
        The model used for generation.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model.
    prompt_fn : callable
        A function that takes a text snippet and returns a formatted prompt.
    temp : float
        Temperature used during generation.
    sample_per_class : int, default 6
        Number of samples to draw per class in each bootstrap iteration.
    n_iterations : int, default 100
        Number of bootstrap iterations.
    batch_size : int, default 4
        Batch size used for inference.
    out_dir : str, default "temp_bootstrap"
        Directory to save intermediate results.

    Returns
    -------
    dict
        A dictionary with metric names as keys and dictionaries containing the mean and 95% confidence intervals.
    """
    # Validate required columns
    required_cols = {"uid", "text", "label"}
    if not required_cols.issubset(set(data.columns)):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Data must contain columns: {missing}")

    # Ensure consistent ordering by resetting index
    data = data.reset_index(drop=True)

    # Perform sampling across iterations
    for seed in range(n_iterations):
        # Sample per class; if sample size > available, sample with replacement
        try:
            sampled = (
                data.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(n=sample_per_class, replace=False, random_state=seed))
            )
        except Exception:
            sampled = (
                data.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(n=sample_per_class, replace=True,random_state=seed))
            )

        sampled = sampled.reset_index(drop=True)

        # Run inference on sampled data with crash recovery
        done = _load_completed_iters(out_dir)
        print(f"Found {len(done)}/{n_iterations} completed iterations in {out_dir}")

        out_fp = os.path.join(out_dir, f"iter_{seed:03d}.json")
        if seed in done:
            continue
        try:
            metrics = run_one_iteration_fn(sampled,
                    model,
                    tokenizer,
                    prompt_fn,
                    temp,
                    batch_size=batch_size) 
            _safe_json_dump({"iter": seed, **metrics}, out_fp)
            print(f"[iter {seed:03d}] saved -> {out_fp}")
        except Exception as e:
            # Save the error so you can inspect it later and still resume
            _safe_json_dump({"iter": seed, "error": repr(e)}, out_fp)
            print(f"[iter {seed:03d}] ERROR saved -> {out_fp}")
            # optionally continue to next iteration (recommended)
            continue

    # --- aggregate at the end ---
    rows = []
    for fp in sorted(glob.glob(os.path.join(out_dir, "iter_*.json"))):
        with open(fp, "r") as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows)
    print(df)

    # drop failed iters
    ok = df[df["error"].isna()] if "error" in df.columns else df
    print(ok)

    # Convert results to DataFrame
    summary = {}
    for m in ["accuracy", "f1_macro", "recall", "precision"]:
        summary[m] = {
            "mean": float(ok[m].mean()),
            "ci_low": ci95(ok[m])[0],
            "ci_high": ci95(ok[m])[1],
            "n_ok": int(ok.shape[0]),
            "n_total": int(df.shape[0]),
        }
    return summary



def run_full_text(model_id, strat_name="chain_of_thought", temp=0.7):
    """Compare model performance across different temperatures using JSON format"""
    print("Running temperature comparison with JSON output format")
    
    # Load data
    data = pd.read_csv("data/set-validation_task-status_label-4_sample-10_desc-stratified.csv")

    # if there is already a 'text' column, remove it to avoid confusion
    if "text" in data.columns:
        data = data.drop(columns=["text"])

    # Rename snippet column to 'text'
    if "Relevant Snippets" in data.columns:
        data = data.rename(columns={"Relevant Snippets": "text"})
    elif "Snippets" in data.columns:
        data = data.rename(columns={"Snippets": "text"})

    # Rename label column to 'label' if present
    if "Label" in data.columns:
        data = data.rename(columns={"Label": "label"})
    elif "label" not in data.columns:
        raise ValueError("Data must contain a 'Label' or 'label' column for ground truth labels.")

    # Ensure a UID column exists for tracking
    if "uid" not in data.columns:
        if "index" in data.columns:
            data = data.rename(columns={"index": "uid"})
        else:
            data['uid'] = range(len(data))

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
    os.makedirs(f"output/status_batch/{model_id}/bootstrap", exist_ok=True)

    
    # Select the prompt function for the specified strategy
    prompt_fn = strategies[strat_name]

    print(f"\nEvaluating strategy: {strat_name}")
    print(f"  Temperature: {temp}")

    # Perform bootstrap evaluation for this model/strategy/temperature
    print("Running bootstrap evaluation for confidence intervals...")
    out_dir = f"output/status_batch/{model_id}/bootstrap"
    ci_stats = bootstrap_evaluation(
        data=data[["uid", "text", "label"]],
        model=model,
        tokenizer=tokenizer,
        prompt_fn=prompt_fn,
        temp=temp,
        sample_per_class=6,
        n_iterations=100,
        batch_size=1,
        out_dir=out_dir
    )

    # Save CI summary to JSON
    ci_output_path = f"{out_dir}/{strat_name}_temp{temp}_bootstrap_summary.json"
    with open(ci_output_path, 'w') as f:
        json.dump(ci_stats, f, indent=2)

    # Print CI results
    print("\nBootstrap confidence intervals (mean, CI lower, CI upper) for each metric:")
    for metric, stats in ci_stats.items():
        print(f"  {metric}: mean={stats['mean']:.4f}, 95% CI=({stats['ci_low']:.4f}, {stats['ci_high']:.4f})")
      

if __name__ == "__main__":
    # Run the temperature comparison with JSON format
    # run_full_text(model_id="medgemma-4b-it", strat_name="structured_reasoning", temp=0.7)
    run_full_text(model_id="gpt-oss-20b", strat_name="structured_reasoning", temp=0.5)
    # run_full_text(model_id="Llama-3.1-8B-Instruct", strat_name="structured_reasoning", temp=0.5)
    print("Done!")