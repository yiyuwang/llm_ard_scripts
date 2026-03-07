import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import string
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import math
from transformers.generation.logits_process import LogitsProcessor


import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from prompts_status import get_prompts



def extract_prediction_from_json(response):
    """Extract the prediction number from JSON response"""
    try:
        # Find JSON pattern in response
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            if 'classification' in data:
                print(f"*****debug**** matched classification in JSON: {data['classification']}")
                return int(data['classification'])
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    
    # Fallbacks if JSON parsing fails
    match = re.search(r"classification\"?\s*[=:]\s*([1-6])", response)
    if match:
        print(f"*****debug**** matched classification pattern: {match.group(1)}")
        return int(match.group(1))
        
    match = re.search(r"([1-4])", response)
    print(f"*****debug**** trying to match simple number pattern")
    
    if match:
        print(f"*****debug**** matched simple number pattern: {match.group(1)}")
        return int(match.group(1))
        
    print(f"Warning: Could not extract prediction from response: {response[:100]}...")
    return 0

def get_model(model_id):
    # Path to your locally downloaded model
    # model_path = f"LLMs/{model_id}"
    # Load tokenizer and model
    if "gemma" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(f"LLMs/{model_id}",
            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            f"LLMs/{model_id}",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
            # Create text generation pipeline
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
        # Create text generation pipeline
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        # llm_pipeline.model = llm_pipeline.model.half()

    else:
        tokenizer = AutoTokenizer.from_pretrained(f"LLMs/{model_id}",
            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            f"LLMs/{model_id}",
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        
        # Create text generation pipeline
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        llm_pipeline.model = llm_pipeline.model.half()

    return llm_pipeline

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
    

def batched_inference(data, model, prompt_fn, temp, batch_size=8, model_id="model", save_dir_base="output/profiler_logs"):
    y_true, y_pred, responses, json_data = [], [], [], []
    
    # Create output directory
    # save_dir = os.path.join(save_dir_base, f"{model_id.replace('/', '_')}_temp{temp}")
    # os.makedirs(save_dir, exist_ok=True)

    total = len(data)
    n_batches = math.ceil(total / batch_size)

    for i in tqdm(range(n_batches), desc=f"Running batches (temp={temp})"):
        batch = data.iloc[i*batch_size:(i+1)*batch_size]
        
        # Preprocess & construct prompts
        texts = [row["text"] for _, row in batch.iterrows()]
        prompts = [prompt_fn(t) for t in texts]
        # medgemma does not work with renormalize_logits
        # if "medgemma" in model_id: # remove renormalize_logits for medgemma because they create gibberish outputs for medgemma
        #     outputs = model(prompts, max_new_tokens=128, temperature=temp, do_sample=True, return_full_text=True,)
        # else:
        #     outputs = model(prompts, max_new_tokens=250, temperature=temp, do_sample=False, renormalize_logits=True, logits_processor=[SanitizeLogits()], return_full_text=True,)
        if temp > 0:
            outputs = model(prompts, max_new_tokens=256, temperature=temp, do_sample=True, return_full_text=True,)
        else:
            outputs = model(prompts, max_new_tokens=256, temperature=temp, do_sample=False, return_full_text=True,)
        print(f"*****debug**** {batch}")
        for j, (_, row) in enumerate(batch.iterrows()):
            print(f"*****debug**** j: {j}")
            response_text = outputs[j][0]["generated_text"]
            pred = extract_prediction_from_json(response_text)

            y_true.append(int(row["label"]))
            y_pred.append(pred)
            responses.append(response_text)

            json_data.append({
                "snippet": texts[j],
                "true_label": int(row["label"]),
                "predicted_label": pred,
                "full_response": response_text
            })


    return y_true, y_pred, responses, json_data


def compare_temperatures_with_json(model_id):
    """Compare model performance across different temperatures using JSON format"""
    print("Running temperature comparison with JSON output format")
    
    # Load data
    data = pd.read_csv("data/set-validation_task-status_label-4_sample-10_desc-stratified.csv")
    # only keep the "Snippets" and "Label" columns
    data = data[["Snippets", "Label"]]
    data = data.rename(columns={"Snippets": "text", "Label": "label"})
    print(data.columns)
    # Get prompt strategies
    strategies = get_prompts()
    
    # Load model
    model = get_model(model_id)

    # Define temperatures to test
    # temperatures = [0.3, 0.5, 0.7, 1]
    temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]
    # Create output directories
    os.makedirs(f"output/status_prompting_final/{model_id}", exist_ok=True)
    os.makedirs(f"output/status_prompting_final/{model_id}/figures", exist_ok=True)
    os.makedirs(f"output/status_prompting_final/{model_id}/prediction", exist_ok=True)

    # Results storage for all strategies and temperatures
    results_by_strategy = {}
    
    # For each strategy
    for strat_name, prompt_fn in strategies.items():
        print(f"\nEvaluating strategy: {strat_name}")
        results_by_strategy[strat_name] = {}
        
        # For each temperature
        for temp in temperatures:
            print(f"  Temperature: {temp}")
            
            # Generate response with current temperature
            y_true, y_pred, responses, json_data = batched_inference(
                    data,
                    model,
                    prompt_fn,
                    temp,
                    batch_size=4,
                    model_id=model_id)
            # Store results
            results_by_strategy[strat_name][temp] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "responses": responses,
                "json_data": json_data
            }
        
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            

            print(f"\n  Temperature {temp} results:")
            print(f"    Accuracy: {acc:.4f}")
            print(f"    F1 score: {f1:.4f}")
            print(f"    Precision (macro): {prec:.4f}")
            print(f"    Recall (macro): {recall:.4f}")


            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4])

            # Store results
            results_by_strategy[strat_name][temp] = {
                "accuracy": acc,
                "f1_macro": f1,
                "recall": recall,
                "precision": prec,
                "confusion_matrix": cm,
                "true_labels": y_true,
                "predicted_labels": y_pred,
                "responses": responses
            }
            
            # Save predictions as JSON
            with open(f"output/status_prompting_final/{model_id}/prediction/{strat_name}_temp{temp}_predictions.json", 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {strat_name} (Temp={temp})')
            plt.savefig(f"output/status_prompting_final/{model_id}/figures/cm_{strat_name}_temp{temp}.png")
            plt.close()
    # Create summary dataframe
    summary_rows = []
    for strat_name in results_by_strategy:
        for temp in temperatures:
            results = results_by_strategy[strat_name][temp]
            summary_rows.append({
                "strategy": strat_name,
                "temperature": temp,
                "accuracy": results["accuracy"],
                "f1_macro": results["f1_macro"],
                "recall": results["recall"],
                "precision": results["precision"],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Create temperature comparison visualizations
    metrics = ["accuracy", "f1_macro", "recall", "precision"]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for strat_name in results_by_strategy:
            # Extract metric values for this strategy across temperatures
            values = [results_by_strategy[strat_name][temp][metric] for temp in temperatures]
            
            # Plot line for this strategy
            plt.plot(temperatures, values, marker='o', linewidth=2, label=strat_name)
            
        plt.xlabel('Temperature')
        plt.ylabel(metric.capitalize())
        plt.title(f'Effect of Temperature on {metric.capitalize()} by Strategy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"output/status_prompting_final/{model_id}/figures/temperature_effect_{metric}.png")
        plt.close()
    
    # Save summary
    summary_df.to_csv(f"output/status_prompting_final/{model_id}/prediction/temperature_strategy_comparison.csv", index=False)

    return summary_df

if __name__ == "__main__":
    # Run the temperature comparison with JSON format
    # summary = compare_temperatures_with_json(model_id="Llama-3.1-8B-Instruct")
    # summary = compare_temperatures_with_json(model_id="medgemma-4b-it")
    summary = compare_temperatures_with_json(model_id="gpt-oss-20b")
    print("\nSummary of results across temperatures:")
    print(summary)