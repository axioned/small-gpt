import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "How many leaves are allowed in 1 year?"

file_path = "../source_documents/handbook.txt"
# Read the text from a file
with open(file_path, 'r', encoding="utf8") as file:
    large_paragraph = file.read()

# Tokenize the question
question_tokens = tokenizer.tokenize(question)

# Initialize a list to store the answers from different segments
answers = []

# Split the large paragraph into smaller segments (you can choose a segment size)
segment_size = 200  # Adjust as needed
segments = [large_paragraph[i:i+segment_size] for i in range(0, len(large_paragraph), segment_size)]

for segment in segments:
    encoding = tokenizer.encode_plus(text=question_tokens, text_pair=segment, add_special_tokens=True, return_tensors='pt', truncation=True)

    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']

    data = model(input_ids=torch.tensor(input_ids), token_type_ids=torch.tensor(token_type_ids))

    start_index = torch.argmax(data['start_logits'])
    end_index = torch.argmax(data['end_logits'])

    answer_tokens = input_ids[0][start_index:(end_index + 1)]
    answer = tokenizer.decode(answer_tokens)
    answers.append(answer)

# Combine answers from different segments
final_answer = " ".join(answers)

print("Final Answer:", final_answer)