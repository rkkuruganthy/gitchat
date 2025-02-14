# local_model.py
from ctransformers import AutoModelForCausalLM

class LocalMistralLLM:
    """
    Local LLM model class using Mistral-7B GGUF with ctransformers.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4):
        """
        Initialize the local Mistral-7B model.

        Parameters:
          model_path (str): Full path to your mistral-7b-instruct-v0.1.Q4_K_M.gguf file.
          n_ctx (int): Context window size.
          n_threads (int): Number of CPU threads to use.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",  # Ensure we specify Mistral model type
            gpu_layers=0,  # Runs fully on CPU (M1/M2 Macs work better this way)
            context_length=n_ctx
        )

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """
        Generate text based on the prompt.

        Parameters:
          prompt (str): The input prompt.
          temperature (float): Sampling temperature.
          max_tokens (int): Maximum number of tokens to generate.

        Returns:
          str: The generated text.
        """
        response = self.model(prompt, temperature=temperature, max_new_tokens=max_tokens)
        return response  # Direct response from Mistral

# For testing the module locally.
if __name__ == "__main__":
    #model_path = "/Users/ravikuruganthy/myApps/models/Llama-3.2-3B-Instruct-F16.gguf"
    model_path = "/Users/ravikuruganthy/myApps/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    llm = LocalMistralLLM(model_path=model_path, n_ctx=4096, n_threads=4)
    
    test_prompt = "Explain retrieval-augmented generation (RAG)."
    print("Prompt:", test_prompt)
    print("Response:", llm.generate(test_prompt))