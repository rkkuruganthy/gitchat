# local_model.py
from ctransformers import AutoModelForCausalLM

class LocalMistralLLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            gpu_layers=0,
            context_length=n_ctx
        )

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        response = self.model(prompt, temperature=temperature, max_new_tokens=max_tokens)
        return response
