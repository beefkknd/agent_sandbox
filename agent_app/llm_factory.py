import os
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# A mapping of model names to their corresponding provider. This allows the
# factory to determine the correct client to instantiate without relying on
# prefixes in the model string.
_MODEL_SOURCE = {
    # Gemini Models
    "gemini-1.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-pro": "gemini",

    # Ollama Models
    "qwen2.5-coder:7b-instruct": "ollama",
    "qwen2.5-coder:latest": "ollama",
    "qwen3-coder:30b": "ollama",
    "gpt-oss:20b": "ollama",
    "qwen2.5-coder:14b": "ollama",
    "qwen3:8b": "ollama",
    "qwen3:14b": "ollama",
    "deepseek-r1:8b": "ollama",
    "deepseek-r1:14b": "ollama",
    "gemma3:27b": "ollama",
    "llama3.2:3b": "ollama",
    "phi4-reasoning:plus": "ollama",
    "qwen3:32b": "ollama",
    "gemma3:27b": "ollama",
    "deepseek-r1:32b": "ollama",
    "phi4:latest": "ollama",
    "mistral-nemo:latest": "ollama",
}


class LLMFactory:
    """A factory for creating and caching language model clients."""

    def __init__(self):
        """Initializes the factory, loading necessary API keys and setting up a cache."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self._cache = {}

    def create_llm(
        self,
        model_name: str,
        **kwargs,
    ) -> BaseChatModel:
        """
        Creates or retrieves from cache a chat model instance.

        The provider is determined by looking up the model_name in the
        _MODEL_SOURCE mapping. If not found, it falls back to parsing the
        provider from the model name (e.g., 'ollama/llama3.1:8b').

        Args:
            model_name: The unified model name (e.g., "gemini-1.5-pro").
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            An instance of a BaseChatModel.

        Raises:
            ValueError: If the provider is unsupported or if required API keys
                        are missing.
        """
        # Create a unique cache key based on model name and sorted kwargs
        cache_key = (model_name, tuple(sorted(kwargs.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Determine the provider and the model name for the client
        provider = _MODEL_SOURCE.get(model_name)
        actual_model = model_name

        if provider is None:
            # Fallback for models not in the lookup table (e.g., "ollama/...")
            if "/" in model_name:
                provider, actual_model = model_name.split("/", 1)
            else:
                # Assume gemini if no provider is specified and not in the map
                provider = "gemini"
        
        if provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            llm = ChatGoogleGenerativeAI(
                model=actual_model,
                google_api_key=self.gemini_api_key,
                **kwargs,
            )
        elif provider == "ollama":
            llm = ChatOllama(
                model=actual_model,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self._cache[cache_key] = llm
        return llm