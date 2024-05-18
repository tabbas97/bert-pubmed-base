from summarizer import Summarizer, TransformerSummarizer

import argparse

parser = argparse.ArgumentParser(description='Summarize text using BERT/adapted')
parser.add_argument('--text_file', type=str, required=True, help='File to summarize')
parser.add_argument('--model', type=str, default='distilbert-base-uncased', help='Model to use for summarization')
parser.add_argument('--adapter', type=str, help='Adapter to use for specific data')
parser.add_argument('--output_file', type=str, help='File to write summary to')
args = parser.parse_args()

# Read text from file
with open(args.text_file, 'r') as f:
    text = f.read()

if not args.adapter:
        model = Summarizer(args.model)
else:
        import transformers
        custom_model = transformers.AutoModelForMaskedLM.from_pretrained(args.model, return_dict=True)
        custom_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        custom_model.load_adapter(args.adapter)
        model = Summarizer(
                custom_model=custom_model,
                custom_tokenizer=custom_tokenizer,
                hidden=[-2, -1],
                hidden_concat=True
        )

bert_raw = model(text)
bert_summary = ''.join(bert_raw)
print(bert_summary)