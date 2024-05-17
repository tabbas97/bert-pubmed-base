from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import transformers
import torch
import collections
import numpy as np
import math

model_checkpoint = "distilbert-base-uncased"

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, return_dict=True)
model.train()

# Sample text
text = "The quick brown fox jumps over the lazy dog. A quick brown fox jumps over the lazy dog."

# Tokenize the text - It is okay to use the same tokenizer for the 
# pubmed version as the one used for the distilbert model since the
# bert models use the wordpiece tokenizer. The wordpiece tokenizer
# allows for subword tokenization as below:
# (e.g. Immunoglobulin ¼> I ##mm ##uno ##g ##lo ##bul ##in) -> Taken from BioBERT paper
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer(text, return_tensors="pt")

# Data Preparation
from datasets import load_dataset
# Dataset
dataset = load_dataset("ccdv/pubmed-summarization")

def tokenize_function(examples):
    # res = tokenizer(examples["article"])
    # print(res.keys())
    # sys.exit(0)
    res = tokenizer(examples["article"], truncation=True, padding="max_length")
    # if tokenizer.is_fast:
    res["word_ids"] = [res.word_ids(i) for i in range(len(res["input_ids"]))]

    res["labels"] = res["input_ids"].copy()
    return res

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

train_size = 500
# train_size = len(dataset["train"]*0.8)
test_size = int(0.1 * train_size)

downsampled_dataset = dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

downsampled_dataset = downsampled_dataset.map(tokenize_function, batched=True, remove_columns=["article", "abstract"])

from transformers import create_optimizer
num_train_epoch = 4
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01
num_train_steps = num_train_epoch * len(downsampled_dataset)

# optimizer, schedule = create_optimizer(learning_rate, num_train_steps, num_warmup_steps=warmup_steps, weight_decay_rate=weight_decay)
optimizer = transformers.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epoch,
    per_device_train_batch_size=8,
    save_steps = train_size // 2,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to="none",
    do_eval=True,
    push_to_hub=True,
    push_to_hub_model_id=f"{model_checkpoint}-finetuned-pubmed-torch-trained-tabbas97",
    push_to_hub_token=tokenizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    # optimizers=optimizer,
)

# Evaluation before training
pre_train_eval_metrics = trainer.evaluate()
print("BEFORE TRAIN : ", pre_train_eval_metrics)
print("Pre-Train perplexity : ", )
print(f"Perplexity: {math.exp(pre_train_eval_metrics.get('eval_loss')):.2f}")

train_metrics = trainer.train()
print("TRAIN METRICS : ", train_metrics)

eval_metrics = trainer.evaluate()
print("EVAL METRICS : ", eval_metrics)

print("Pre-Train perplexity : ", )
print(f"Perplexity: {math.exp(pre_train_eval_metrics.get('eval_loss')):.2f}")
print("Post-Train perplexity : ", )
print(f"Perplexity: {math.exp(eval_metrics.get('eval_loss')):.2f}")

# Save the model
model.save_pretrained(f"{model_checkpoint}-finetuned-pubmed-torch-trained-tabbas97")
tokenizer.save_pretrained(f"{model_checkpoint}-finetuned-pubmed-torch-trained-tabbas97")

# Push to HfHub
trainer.push_to_hub()

# Load the model
model_reload = AutoModelForMaskedLM.from_pretrained(f"{model_checkpoint}-finetuned-pubmed-torch-trained-tabbas97")
model_reload.eval()

# Inference
text = "The quick brown fox jumps over the lazy dog. A quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
outputs = model_reload(**inputs)
logits = outputs.logits
predicted_index = torch.argmax(logits, dim=-1)

predicted_text = tokenizer.decode(predicted_index[0])

print(predicted_text)


