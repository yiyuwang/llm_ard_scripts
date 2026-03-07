import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import string
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import re, json, math, time
from transformers.generation.logits_process import LogitsProcessor
import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from textwrap import wrap
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    tokenizer = get_tokenizer("LLMs/gatortron-base")
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["label"] for b in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


def create_data_loader(df, tokenizer, max_len, batch_size, collate_fn=collate_fn):
    ds = CannabisClassData(
        text=df.text.to_numpy(),
        label=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
        )
     
class CannabisClassifierFrozenBackbone(nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, n_classes, dropout_rate=0.1):
        super(CannabisClassifierFrozenBackbone, self).__init__()
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.drop=nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output=self.drop(pooled_output)
        output=self.out(output)
        #return self.softmax(output)
        return nn.LogSoftmax(dim=1)(output)


class CannabisClassData(Dataset):
  def __init__(self, text, label, tokenizer, max_len):
    self.text=text
    self.label=label
    self.tokenizer=tokenizer
    self.max_len=max_len
    
  def __len__(self):
    return len(self.text)
  
  def __getitem__(self,item):
    text= str(self.text[item])
    label=self.label[item]
    encoding=self.tokenizer.encode_plus(
    text,
    max_length=self.max_len,
    add_special_tokens=True,
    # pad_to_max_length=True,
    truncation =True,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors='pt'
    )
    return{
        'text':text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'label':torch.tensor(label,dtype=torch.long)
    }

    

class CannabisClassifier(nn.Module):
    def __init__(self,PRE_TRAINED_MODEL_NAME,n_classes, dropout_rate=0.3):
        super(CannabisClassifier,self).__init__()
        self.bert=AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME, local_files_only=True)
        self.drop=nn.Dropout(p=dropout_rate)
        self.out=nn.Linear(self.bert.config.hidden_size,n_classes)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,input_ids,attention_mask):
        _,pooled_output=self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=False
        )
        output=self.drop(pooled_output)
        output=self.out(output)
        #return self.softmax(output)
        return nn.LogSoftmax(dim=1)(output)
    
def get_tokenizer(PRE_TRAINED_MODEL_dir):
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_dir, local_files_only=True)
    return tokenizer


def get_model(params, df_test, n_classes=None):
    PRE_TRAINED_MODEL_dir = "LLMs/gatortron-base"
    best_state_path = f"output/gatortron-base_finetune_reason/best_model/best_model_state_tune-{params['tune_mode']}.bin"
    best_state = torch.load(best_state_path, map_location=device)
    # — rebuild best model (train + val) —————————————————
    if n_classes is not None:
        num_classes = n_classes
    else:
        num_classes = len(np.unique(df_test.label))
    if params.get("tune_mode", "full") == "full":
        model = CannabisClassifier(
            PRE_TRAINED_MODEL_NAME=PRE_TRAINED_MODEL_dir,
            n_classes=num_classes,
            dropout_rate=params["dropout"],
        )
    else:
        model = CannabisClassifierFrozenBackbone(
            PRE_TRAINED_MODEL_NAME=PRE_TRAINED_MODEL_dir,
            n_classes=num_classes,
            dropout_rate=params["dropout"],
        )

    model.load_state_dict(best_state)
    return model

import torch

def eval_model(model, data_loader, loss_fn, device):
    model.eval()  # set model to evaluation mode
    model.to(device)
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # predictions = argmax over class dimension
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

def get_predictions(model, data_loader, device):
    model = model.eval()
    model.to(device)

    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            
            real_values.extend(label)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, prediction_probs, real_values


def bootstrap_evaluation(
    data: pd.DataFrame,
    model: nn.Module,
    tokenizer,
    params: Dict,
    device,
    sample_per_class: int = 6,
    n_iterations: int = 100,
    base_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Perform bootstrap sampling to estimate confidence intervals for classification metrics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'uid', 'text', and 'label' columns.
    model : nn.Module
        The fine-tuned GatorTron model for classification.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to create input sequences.
    params : dict
        Dictionary containing 'max_len' and 'batch' for DataLoader creation.
    device : torch.device
        Torch device for inference.
    sample_per_class : int, default 6
        Number of samples to draw from each class in each bootstrap iteration.
    n_iterations : int, default 100
        Number of bootstrap iterations.
    base_seed : int, default 42
        Base seed for reproducibility; each iteration uses base_seed + i.

    Returns
    -------
    dict
        A dictionary with metric names as keys and dictionaries containing the mean and 95% confidence intervals.
    """
    # Ensure required columns exist
    required_cols = {"uid", "text", "label"}
    if not required_cols.issubset(set(data.columns)):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Data must contain columns {missing}")

    metrics_list = []
    # Reset index for sampling consistency
    data = data.reset_index(drop=True)

    # Put model in evaluation mode and on device
    model = model.eval().to(device)

    for i in range(n_iterations):
        seed = base_seed + i

        # Sample per class; use replacement if group smaller than sample size
        def sample_group(x):
            replace_flag = True if len(x) < sample_per_class else False
            return x.sample(n=sample_per_class, replace=replace_flag, random_state=seed)

        sampled = data.groupby("label", group_keys=False).apply(sample_group)
        sampled = sampled.reset_index(drop=True)

        # Create DataLoader for sampled data
        sample_loader = create_data_loader(sampled, tokenizer, params['max_len'], 1)

        # Run inference
        y_pred, _, y_true = get_predictions(model, sample_loader, device)
        # Convert tensors to lists
        y_true_list = y_true.numpy().tolist()
        y_pred_list = y_pred.numpy().tolist()

        # Compute metrics
        acc = accuracy_score(y_true_list, y_pred_list)
        f1 = f1_score(y_true_list, y_pred_list, average="macro")
        rec = recall_score(y_true_list, y_pred_list, average="macro")
        prec = precision_score(y_true_list, y_pred_list, average="macro")

        metrics_list.append({
            "accuracy": acc,
            "f1_macro": f1,
            "recall": rec,
            "precision": prec,
        })

    metrics_df = pd.DataFrame(metrics_list)
    summary_stats = {}
    for metric in ["accuracy", "f1_macro", "recall", "precision"]:
        values = metrics_df[metric]
        mean_val = values.mean()
        lower = values.quantile(0.025)
        upper = values.quantile(0.975)
        summary_stats[metric] = {
            "mean": mean_val,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    return summary_stats



def run_full_text(params, device):
    """use fine tuned gatortron to extract information from full text"""
    
    # Load data
    data = pd.read_csv("data/set-validation_task-reason_label-6_sample-10_desc-stratified.csv")

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
    
    # create uid column if not present
    # if "uid" not in data.columns:
    #     if "index" in data.columns:
    #         data = data.rename(columns={"index": "uid"})
    #     else:
    data['uid'] = range(len(data))

    data['label'] = data['label']-1
    

    # Print columns for debugging
    print(data.columns)
    
    # Load model
    PRETRAINED_MODEL_dir = "LLMs/gatortron-base"
    tokenizer = get_tokenizer(PRETRAINED_MODEL_dir)
    # Determine number of classes from data
    num_classes = int(data["label"].nunique())
    model = get_model(params, data, num_classes)
    
    # Inference on the full dataset
    model.eval()
    test_loader = create_data_loader(data, tokenizer, params['max_len'], 1)

    # save predicted labels
    model.to(device)
    y_pred, _, y_test = get_predictions(
        model,
        test_loader,
        device,
    )


    # Create output directories
    # save y_pred and y_test: y_test contains the true labels
    tdir = f"output/gatortron-base_finetune_reason/bootstrap/"
    os.makedirs(tdir, exist_ok=True)
    # predictions = []
    # for i in range(len(y_test)):
    #     predictions.append({
    #         "uid": int(data.iloc[i]["uid"]),
    #         "true_label": int(y_test[i].item()),
    #         "predicted_label": int(y_pred[i].item()),
    #     })
    # with open(os.path.join(tdir, "fine-tuned_GatorTron_predictions.json"), "w") as f:
    #     json.dump(predictions, f)

    # Run bootstrap evaluation for this model
    print("Running bootstrap evaluation for confidence intervals...")
    ci_stats = bootstrap_evaluation(
        data=data[['uid', "text", "label"]],
        model=model,
        tokenizer=tokenizer,
        params=params,
        device=device,
        sample_per_class=6,
        n_iterations=100,
        base_seed=42,
    )

    # Save CI summary to JSON
    ci_output_path = os.path.join(tdir, "fine-tuned_GatorTron_bootstrap_summary.json")
    with open(ci_output_path, 'w') as f:
        json.dump(ci_stats, f, indent=2)

    # Print CI results
    print("\nBootstrap confidence intervals (mean, CI lower, CI upper) for each metric:")
    for metric, stats in ci_stats.items():
        print(f"  {metric}: mean={stats['mean']:.4f}, 95% CI=({stats['ci_lower']:.4f}, {stats['ci_upper']:.4f})")

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = json.load(open("output/gatortron-base_finetune_reason/best_model/best_model_params.json"))
    print(params)
    run_full_text(params, device)
    print("Done!")
