# PEFT Fine-tune of DistilBERT on Pubmed

This is a finetune of DistilBERT finetuned to run on Pubmed dataset. The model is finetune on a L4 GPU.

The pubmed-torch version was fully finetuned on the Pubmed dataset. The lora version is only an adapter that is finetuned on the dataset. We also only lora tuned the query and value weights of the attention layer. This was the suggestion in the original LORA paper.

## In Progress
- [] Fixing summarizer module to allow lora adapted model.

## Completed

- [x] Fully finetuned model
- [x] LORA finetuned model
- [x] Training and evaluation scripts
- [x] Model cards - Auto pushed as TrainingCallbacks
- [x] TF version of the model
- [x] PEFT Early stopping

## Future TODOs

- [ ] Add more datasets
- [ ] Add more models
- [ ] Experiment with parameter sweep on rank and alpha values
- [ ] Experiment with different layers to LORA tune

Model Cards:

- Fully finetuned model: <https://huggingface.co/tabbas97/distilbert-base-uncased-finetuned-pubmed-torch-trained-tabbas97>
- LORA finetuned model: <https://huggingface.co/tabbas97/distilbert-base-uncased-finetuned-pubmed-lora-trained-tabbas97>
- TF fully finetuned model: <https://huggingface.co/tabbas97/distilbert-base-uncased-finetuned-pubmed-tabbas97>

The TF version uses TF specfic training and evaluation modules. The torch version uses the generic Trainer and TrainingArguments from transformers.
