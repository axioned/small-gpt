from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-cased-distilled-squad'
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Context and question
file_path = "./ingest/handbook_full.txt"
with open(file_path, encoding='utf8') as f:
    context = f.read()

# Question
question = "What is the main topic of this document?"

# Tokenize the question
question_tokens = tokenizer(question, return_tensors="pt")

# Initialize variables to store answers
answers = []

# Tokenize and process the document in chunks
max_chunk_length = 512  # Set the chunk length based on the model's maximum token limit
chunks = [context[i:i + max_chunk_length] for i in range(0, len(context), max_chunk_length)]

for chunk in chunks:
    # Tokenize the chunk
    chunk_tokens = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_chunk_length, padding='max_length')

    print(chunk_tokens)

    # Get the answer
    start_scores, end_scores = model(**chunk_tokens)

    # Find the answer span
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1  # Adding 1 to include the end token

    # Extract the answer from the chunk
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(chunk_tokens.input_ids[0][start_index:end_index]))

    answers.append(answer)

# Combine answers from different chunks
final_answer = " ".join(answers)

print("Question:", question)
print("Answer:", final_answer)
