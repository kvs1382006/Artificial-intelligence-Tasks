from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_paragraph(prompt_text: str, max_length: int = 150) -> str:
    """
    Generates coherent paragraphs using pre-trained GPT-2 based on a user prompt.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Encode prompt
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")
    
    # Generate text output sequence
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    user_prompt = "The future of artificial intelligence in space exploration holds immense potential,"
    
    print("=" * 60)
    print(f"USER PROMPT: '{user_prompt}'")
    print("=" * 60)
    
    result = generate_paragraph(user_prompt)
    print("\nGENERATED OUTPUT:")
    print(result)