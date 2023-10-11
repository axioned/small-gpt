import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

file_path = "./ingest/leaves.txt"
with open(file_path, encoding='utf8') as f:
    text = f.read()
    
# tokenized_lines = [tokenizer.encode(line) for line in lines]

# print(text)
q2 = "Can trainees take leave ?"
e2 = tokenizer.encode_plus(text=q2, text_pair=text, add_special_tokens=True, return_tensors='pt')

input_ids_2 = e2['input_ids']  # Token embeddings
token_type_ids_2 = e2['token_type_ids']  # Segment embeddings

data2 = model(input_ids=torch.tensor(input_ids_2), token_type_ids=torch.tensor(token_type_ids_2))

start_index_2 = torch.argmax(data2['start_logits'])
end_index_2 = torch.argmax(data2['end_logits'])

answer_tokens_2 = input_ids_2[0][start_index_2:(end_index_2 + 1)]
answer_2 = tokenizer.decode(answer_tokens_2)

print("Answer:", answer_2)

