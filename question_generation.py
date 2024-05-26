import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast

# Set Hugging Face token for authentication
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Clear GPU cache
torch.cuda.empty_cache()

# Load the Mistral 7B model and tokenizer with authentication
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', token=os.getenv("HUGGINGFACE_TOKEN"))
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', device_map='cuda', token=os.getenv("HUGGINGFACE_TOKEN"))

print("Model loaded.")

# Adjust settings as needed
def generate_question_and_answer(text, max_new_tokens=200, temperature=0.7, top_p=0.9, top_k=50):
    # Prepare the input text
    input_text = f"Read the following passage and generate a relevant question and answer based on its content:\n\n{text}\n\nQuestion:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=4096).to('cuda')
    
    # Generate the question and answer with mixed precision
    with torch.no_grad():
        with autocast():
            output_ids = model.generate(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2, 
                top_p=top_p, 
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id to eos_token_id
            )
   
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract the question and answer part (if provided)
    question_start = generated_text.find("Question:") + len("Question:")
    question_end = generated_text.find("Answer:", question_start)
    question = generated_text[question_start:question_end].strip()
    answer_start = question_end + len("Answer:")
    answer = generated_text[answer_start:].strip()
    
    return question, answer

# Directory containing the chunked documents
chunked_documents_dir = 'path/to/output_directory'
chunk_sizes = [128, 256, 512, 1024, 2048]

# Process each chunk size separately
for size in chunk_sizes:
    size_dir = os.path.join(chunked_documents_dir, str(size))
    
    # Create a CSV file for the current chunk size
    output_csv = f'generated_questions_{size}.csv'
    
    # Open the CSV file in write mode
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Chunk Size', 'Generated Question', 'Generated Answer'])
        
        # Process each chunk one by one
        for filename in os.listdir(size_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(size_dir, filename), 'r', encoding='utf-8') as file:
                    chunk = file.read()
                
                try:
                    question, answer = generate_question_and_answer(chunk)
                    writer.writerow([filename, size, question, answer])
                    print(f"Generated question and answer for {filename} with chunk size {size}")
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error generating question and answer for {filename}: {e}")

    print(f"Generated questions and answers saved to {output_csv}.")

