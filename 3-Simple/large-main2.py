from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Context and question
file_path = "./ingest/handbook_full.txt"
with open(file_path, encoding='utf8') as f:
    context = f.read()

# question = "What is the speed of car?"
# question = "What time Sarah will reach halfway point?"
# question = "What is Axioned Handbook?"
# question = "Who is the CFO?"
# question = "What is the email address for PMO?"
# question = "WHo is sandip?"
# question = "who is tim ?"
# question = "What does Axioned serves ?" # fails
question = "What is SME ?"

# Tokenize the input
inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding='max_length')

# Get the answer
data = model(**inputs)


# Convert token indices to actual tokens
# start_index = torch.argmax(data['start_logits']).item()
# end_index = torch.argmax(data['end_logits']).item()

########## V1 
# print(start_index)
# start_token = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index])
# end_token = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][end_index])
# Join tokens to form the answer
# answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))


########## V2 
# Check if the answer is not empty
# if start_index < end_index:
#     start_token = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index])
#     end_token = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][end_index])
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index + 1]))
# else:
#     answer = "Answer not found"

######### V3 

# Handle overflowing tokens
if "overflowing_tokens" in inputs:
    overflowing_tokens = inputs["overflowing_tokens"]
    if len(overflowing_tokens) > 0:
        # If there are overflowing tokens, you may choose to process them separately or handle them as needed.
        # For simplicity, this example assumes that the first sequence of overflowing tokens is relevant.
        data = model(input_ids=overflowing_tokens[0].unsqueeze(0)) 


# Convert start and end indices to Python integers
start_index = torch.argmax(data["start_logits"]).item()
end_index = torch.argmax(data["end_logits"]).item()

print(inputs["input_ids"][0][start_index].tolist())

# Get the tokens
start_token = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index].tolist())
end_token = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][end_index].tolist())

# Join tokens to form the answer
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index + 1]))


print("Question:", question)
print("Answer:", answer)
