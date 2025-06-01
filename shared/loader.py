from transformers import BertForMaskedLM, BertTokenizer

model_name = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()
