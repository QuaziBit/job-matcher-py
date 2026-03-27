# ollama_utils.py
import math

# Approximate context windows for your models
MODEL_CONTEXT_WINDOWS = {
    "llama3.1:8b": 8192,
    "gemma3:27b": 32768,
    "nemotron-3-nano:latest": 16384,
}

def estimate_tokens(text: str) -> int:
    """
    Rough token estimator: 1 token ≈ 4 characters
    """
    return math.ceil(len(text) / 4)

def safe_num_predict(prompt: str, model_name: str, desired_output: int = 4096) -> int:
    """
    Calculate a safe num_predict for Ollama.
    
    Parameters:
    - prompt: combined system + user messages as a single string
    - model_name: the model you are using (must be in MODEL_CONTEXT_WINDOWS)
    - desired_output: desired number of output tokens

    Returns:
    - num_predict: safe number of tokens for Ollama generation
    """
    try:
        context_window = MODEL_CONTEXT_WINDOWS.get(model_name)
        if context_window is None:
            raise ValueError(f"Unknown model '{model_name}'. Add it to MODEL_CONTEXT_WINDOWS.")
        
        prompt_tokens = estimate_tokens(prompt)
        max_safe_output = context_window - prompt_tokens

        if max_safe_output <= 0:
            raise ValueError("Prompt is too long for the model's context window!")

        return min(desired_output, max_safe_output)

    except Exception as e:
        # Optionally, log the error here if you have a logger
        # print(f"[ollama_utils] safe_num_predict error: {e}")
        # Fallback: return a conservative default
        print(f"\n[IN ollama_utils.py] safe_num_predict error: {e}\n")
        return min(desired_output, 1024)