from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint

my_checkpoint = "chsubhasis/ai-tutor-towardsai-updated"
loaded_model = AutoModelForCausalLM.from_pretrained(my_checkpoint)
loaded_tokenizer = AutoTokenizer.from_pretrained(my_checkpoint)

def generate_response_direct(model, tokenizer, prompt, max_length=500):
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    # Decode and return response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response_HF(prompt):
    llm = HuggingFaceEndpoint(
        repo_id="chsubhasis/ai-tutor-towardsai-updated",
        task="question-answering",
        max_new_tokens= 100,
        do_sample= True,
        temperature = 0.7,
        repetition_penalty= 1.2
    )
    return llm.invoke(prompt) 

if __name__ == "__main__":
    #print(generate_response_direct(model=loaded_model, tokenizer=loaded_tokenizer, prompt="What is Artificial Intelligence?"))
    print(generate_response_HF(prompt="What is Artificial Intelligence?"))