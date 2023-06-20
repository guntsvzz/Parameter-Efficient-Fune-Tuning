#comment this if you are not using AIT proxy...
import os
os.environ['http_proxy']  = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

from datasets import load_dataset, load_metric

###1. Load Dataset
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
task_name = "wnli"
datasets = load_dataset("glue", task_name)
datasets["train"][3]

label_names = datasets['train'].features['label'].names
id2label = {str(i): label for i, label in enumerate(label_names)}

import numpy as np 
num_labels = np.unique(datasets['train']['label']).size
num_labels

###2.Preprocessing
# Labels
if task_name is not None:
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
else:
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        
from transformers import AutoModelForSequenceClassification, PretrainedConfig

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and task_name is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        
elif task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    args = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    result = tokenizer(*args, max_length=180, padding="max_length", truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

    return result

tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(list(task_to_keys[task_name]) + ["idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42) #.select(range(100))
small_eval_dataset = tokenized_datasets["validation_matched" if task_name == "mnli" else "validation"].shuffle(seed=42) #.select(range(100))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42)

###3. Data Loader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)
# small_train_dataset['labels'].unique(), small_eval_dataset['labels'].unique()

###4. Model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Original parameters :', count_parameters(model))

###5. Training
from torch.optim import AdamW
##optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
from transformers import get_scheduler
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)

##Accelerate
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
### Cuda Checking
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    TaskType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    LoraConfig,
)
PEFTtechnique = 'FT'
if PEFTtechnique == 'FT':
    pass
elif PEFTtechnique == 'Adapter':
    from transformers.adapters import BertAdapterModel, AutoAdapterModel 
    model = BertAdapterModel.from_pretrained("bert-base-cased", num_labels=num_labels) 
    # Add a new adapter
    model.add_adapter(task_name)
    # Add a matching classification head
    model.add_classification_head(
        task_name,
        num_labels=num_labels,
        id2label=id2label
      )
    # Activate the adapter
    model.train_adapter(task_name)
elif PEFTtechnique == 'Prefix':
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 
elif PEFTtechnique == 'Prompt':
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 
elif PEFTtechnique == 'LoRA':
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 
elif PEFTtechnique == 'BitFit':
    # Freeze all parameters except biases
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param.requires_grad = False 

###Metric
import numpy as np
import evaluate
metric = evaluate.load("accuracy")
# metric = load_metric("glue", task_name)

####5. Training
import torch
from tqdm.auto import tqdm
gradient_accumulation_steps = 1
progress_bar = tqdm(range(num_training_steps))
model.train()
eval_metrics = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss/gradient_accumulation_steps
        # loss.backward()
        accelerator.backward(loss)
        # Step with optimizer
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    model.eval()
    for batch in eval_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(
            predictions=predictions, 
            references=batch["labels"])
        
    eval_metric = metric.compute()
    eval_metrics += eval_metric.values()
    print(f"Epoch at {epoch}: {eval_metric}")
print('Avg Metric', eval_metrics/len(eval_metrics))