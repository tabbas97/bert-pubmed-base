from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import transformers
import tensorflow as tf
import collections
import numpy as np

model_checkpoint = "distilbert-base-uncased"

model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Sample text
text = "The quick brown fox jumps over the lazy dog. A quick brown fox jumps over the lazy dog."

# Tokenize the text
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer(text, return_tensors="pt")


# Data Preparation
from datasets import load_dataset
# Dataset
dataset = load_dataset("ccdv/pubmed-summarization")

# print(dataset)

# Full dataset
FULL_DATASET = False

data = dataset["train"]
if not FULL_DATASET:
    data = data.shuffle(seed=42).select(range(20000))

# We only need the train set - and only the article text itself
# data = dataset["train"]["article"]
import sys

def tokenize_function(examples):
    # res = tokenizer(examples["article"])
    # print(res.keys())
    # sys.exit(0)
    res = tokenizer(examples["article"], truncation=True, padding="max_length")
    # if tokenizer.is_fast:
    res["word_ids"] = [res.word_ids(i) for i in range(len(res["input_ids"]))]

    res["labels"] = res["input_ids"].copy()
    return res

# Leaving in all columns for now
# tokenized_dataset = data.map(tokenize_function, batched=True, remove_columns=["article", "abstract"])

# We are not using the concat and chunk strategy here. We might need to do that for 
# the tasks where the input is consistently less than 512 tokens (input_length).


from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.data.data_collator import tf_default_data_collator

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    # print(features)
    for feature in features:
        # print(feature)
        print(feature.keys())
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return tf_default_data_collator(features)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# samples =  [tokenized_dataset[i] for i in range(2)]
# print(samples)
# for sample in samples:
#     _ = sample.pop("word_ids")

# for chunk in data_collator(samples)["input_ids"]:
#     print(tokenizer.decode(chunk))

train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

downsampled_dataset = downsampled_dataset.map(tokenize_function, batched=True, remove_columns=["article", "abstract"])

samples = [downsampled_dataset["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

print(downsampled_dataset)

tf_train_dataset = model.prepare_tf_dataset(
    downsampled_dataset["train"],
    collate_fn=data_collator,
    # collate_fn_args = {"word_ids":"word_ids"},
    shuffle=True,
    batch_size=32,
)

tf_eval_dataset = model.prepare_tf_dataset(
    downsampled_dataset["test"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)


from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model_name = model_checkpoint.split("/")[-1]
callback = PushToHubCallback(
    output_dir=f"{model_name}-finetuned-pubmed-tabbas97", tokenizer=tokenizer
)

import math

eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])

eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")