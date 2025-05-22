# clients/google_client.py
import time
from typing import Dict, Any, Tuple, Optional
import google.generativeai as genai
from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class GoogleClient(BaseClient):
    """Client for Google models (Gemini)"""

    def __init__(self, model: str, config: Dict[str, Any]):
        super().__init__(model, config)
        genai.configure(api_key=self.api_key)
        self.model_obj = genai.GenerativeModel(model_name=self.model)

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with Google models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Configure generation parameters
            generation_config = {
                "max_output_tokens": params.get("max_tokens", 1000),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
                "top_k": params.get("top_k", 40)
            }

            # Make API call
            response = self.model_obj.generate_content(
                prompt,
                generation_config=generation_config
            )

            # End timing
            elapsed_time = time.time() - start_time

            # Process response
            content = response.text

            # Estimate token usage (Google doesn't provide exact counts)
            input_tokens = len(prompt) // 4  # Very rough estimate
            output_tokens = len(content) // 4  # Very rough estimate

            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "elapsed_time": elapsed_time
            }

            result = {
                "success": True,
                "content": content,
                "model": self.model,
                "provider": "google"
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from Google model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            # Configure generation parameters
            generation_config = {
                "max_output_tokens": params.get("max_tokens", 1000),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
                "top_k": params.get("top_k", 40)
            }

            response = self.model_obj.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with {self.model}: {e}")
            raise