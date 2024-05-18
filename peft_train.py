from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import math
from transformers import EarlyStoppingCallback
# from transformers.integrations import TrainerCallback
# model_name_or_path = "./distilbert-base-uncased-finetuned-pubmed-torch-trained-tabbas97/"
# tokenizer_name_or_path = "./distilbert-base-uncased-finetuned-pubmed-torch-trained-tabbas97/"

model_name_or_path = "distilbert-base-uncased"

# Loading the base model and tokenizer to demonstrate the training
# speed difference in using PEFT.
model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# model.print_trainable_parameters()

# The downstream task is intended to be feature extraction to be used as embeddings
# in turn to be used for clustering task. ---> FEATURE EXTRACTION TASK
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, 
    init_lora_weights="gaussian", # -> might choose pissa
    r = 16, # -> might choose 32
    lora_alpha = 32, # General recommendation is r * 2. Higher ranks can allow for smaller alpha/r ratios
    target_modules=["q_lin", "v_lin"],  # In general, we want to target the query and value matrices. 
                                # Suggested in the original paper. But can help with any other linear / large kernel matrices.
    lora_dropout = 0.05,
    bias = "none",
)

# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Preparing the dataset
from datasets import load_dataset
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

# train_size = 10000
train_size = int(len(dataset["train"])*0.8)
test_size = int(0.1 * train_size)

downsampled_dataset = dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

downsampled_dataset = downsampled_dataset.map(tokenize_function, batched=True, remove_columns=["article", "abstract"])

num_train_epoch = 4
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01
num_train_steps = num_train_epoch * len(downsampled_dataset)

training_args = TrainingArguments(
    output_dir="./results-lora",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epoch,
    per_device_train_batch_size=32,
    # save_steps = train_size // 2,
    save_steps = 2000,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to="none",
    do_eval=True,
    push_to_hub=True,
    push_to_hub_model_id=f"{model_name_or_path}-finetuned-pubmed-lora-trained-tabbas97",
    push_to_hub_token=tokenizer,
    label_names=["labels"], # This is due to the PEFT not yet being 
                            # compatible with the internal find_labels function
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,
    metric_for_best_model='eval_loss'
)

# Defining Custom Callback to calculate the perplexity during training

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
    ]
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
model.save_pretrained(f"{model_name_or_path}-finetuned-pubmed-lora-trained-tabbas97")
tokenizer.save_pretrained(f"{model_name_or_path}-finetuned-pubmed-lora-trained-tabbas97")

# Push to HfHub
trainer.push_to_hub()

# Load the model
model_reload = AutoModelForMaskedLM.from_pretrained(f"{model_name_or_path}-finetuned-pubmed-lora-trained-tabbas97")
model_reload.eval()