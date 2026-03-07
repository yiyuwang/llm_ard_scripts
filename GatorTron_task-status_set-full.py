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
    best_state_path = f"output/gatortron-base_finetune_status/best_model/best_model_state_tune-{params['tune_mode']}.bin"
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



def run_full_text(params, device):
    """use fine tuned gatortron to extract information from full text"""
    
    # Load data
    data = pd.read_csv("data/matched_rows_with_context_buffer-100_uid.csv")
    data = data.rename(columns={"Relevant Snippets": "text"})

    # code requires 'text' and 'label' columns
    # use uid as labels for future matching
    data['label'] = data['uid'].astype(int)
    print(data.columns)
    
    # Load model
    PRETRAINED_MODEL_dir = "LLMs/gatortron-base"
    tokenizer = get_tokenizer(PRETRAINED_MODEL_dir)
    model = get_model(params,data, 4)
    
    # Create output directories
    # get the date for model version
    today = time.strftime("%Y%m%d")
    model_id = f"gatortron-base_finetune_status_{today}"
    os.makedirs(f"output/status_batch/{model_id}", exist_ok=True)


    # Inference
    model.eval()
    loss_fn=nn.CrossEntropyLoss().to(device)
    # — final **test** evaluation ————————————————————
    test_loader = create_data_loader(data, tokenizer, params['max_len'], params['batch'])
    

    # save predicted labels
    model.to(device)
    y_pred, _, y_test = get_predictions(
        model,
        test_loader,
        device,
    )
    # save y_pred and y_test
    # y_test is uid in this case, so we can match back to original data
    # save in json format for standardized extraction
    tdir = f"output/status_batch/{model_id}/"
    predictions = []
    for i in range(len(y_test)):
        predictions.append({
            "uid": int(y_test[i].item()),
            "predicted_label": int(y_pred[i].item()),
        })
    with open(os.path.join(tdir, "fine-tuned_GatorTron_predictions.json"), "w") as f:
        json.dump(predictions, f)

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = json.load(open("output/gatortron-base_finetune_status/best_model/best_model_params.json"))
    run_full_text(params, device)
    print("Done!")
