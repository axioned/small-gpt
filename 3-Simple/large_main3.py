from transformers import  BertTokenizer, BertForQuestionAnswering
import torch

# Load the SpanBERT model and tokenizer
model_path = "./models/spanbert"
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# Context and question
file_path = "./ingest/handbook_full.txt"
with open(file_path, encoding='utf8') as f:
    context = f.read()

question = "What does Axioned serves ?"

# Tokenize the input
inputs = tokenizer(question, context, return_tensors="pt")

# Get the answer
data = model(**inputs)

# Find the answer span
start_index = torch.argmax(data['start_logits'])
end_index = torch.argmax(data['end_logits'])

# Extract the answer from the document
answer = tokenizer.decode(inputs.input_ids[0][start_index:end_index + 1])

print("Question:", question)
print("Answer:", answer)
