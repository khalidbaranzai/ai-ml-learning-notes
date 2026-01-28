# Import required libraries from Hugging Face transformers
from transformers import pipeline, AutoTokenizer

# Function to create and initialize a simple language model
def create_simple_llm():
    
    model_name = "distilgpt2"
    
    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)
    
    return generator

# generator = create_simple_llm()

# prompt = "Brahvi is a "
# generated_text = generator(prompt, max_length=100, num_return_sequences=1)

# print(generated_text[0]['generated_text'])


def generate_text(generator, prompt, max_length=100):
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    
    return result[0]['generated_text']


def run_llm_demo():
    generator = create_simple_llm()
    """
    Demonistrates basic LLM functionality with explantions
    """
    
    print("Loading Simple LLM Model...")
    print("Simple LLM Demo ")
    print("This demo shows basic text generation using a small language")
    
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Python programming is",
    ]
    
    for prompt in prompts:
        print("\nPrompt: ", prompt)
        print("\nGenerated Text: ", generate_text(generator, prompt))
        input("\nPress enter to see next example: ")
        
        

def interactive_demo():
    """
    Allows user to interact with Model
    """
    
    generator = create_simple_llm()
    print("\nInteractive LLM Demo ")
    print("\nType your prompts or 'Quit' to exit: ")
    
    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() == 'quit':
            break
        
        response = generate_text(generator, prompt)
        print('\n💬Generated response: ')
        print(response)
    



def explain_process():
    print("How it works:")
    print("1. Input text -> Tokenization -> Numbers")
    print("2. Numbers -> Model Processing -> Prediction")
    print("3. Prediction -> New Token -> Output Text")
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Hello World!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print("Example Tokenization: ")
    print(f"Original Text: {text}")
    print(f"As tokens (numbers): {tokens}")
    print(f"Decoded text: {decoded}")
    
    
if __name__ == "__main__":
    print("Choose a choice: ")
    print("1. Run basic demonistration")
    print("2. Interactive mode")
    print("3. Explain the process")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        run_llm_demo()
    elif choice == '2':
        interactive_demo()
    elif choice == '3':
        explain_process()
    else:
        print("Invalid choice!")
        
    
    
    