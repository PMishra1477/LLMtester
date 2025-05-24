# clients/mistral_client.py
import time
import requests
from typing import Dict, Any, Tuple, Optional
from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class MistralClient(BaseClient):
    """Client for Mistral AI models"""

    def __init__(self, model: str, config: Dict[str, Any]):
        super().__init__(model, config)
        self.api_url = "https://api.mistral.ai/v1/chat/completions"

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with Mistral models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": params.get("max_tokens", 1000),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0)
            }

            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                self.api_url,
                json=request_data,
                headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            # End timing
            elapsed_time = time.time() - start_time

            # Process response
            content = response_data["choices"][0]["message"]["content"]

            # Extract usage metrics
            usage = self._extract_usage(response_data)
            usage["elapsed_time"] = elapsed_time

            result = {
                "success": True,
                "content": content,
                "model": self.model,
                "provider": "mistral"
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from Mistral model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": params.get("max_tokens", 1000),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0)
            }

            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                self.api_url,
                json=request_data,
                headers=headers
            )

            response.raise_for_status()
            response_data = response.json()
            
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating text with {self.model}: {e}")
            raise

    def _extract_usage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token usage from API response"""
        usage_data = response_data.get("usage", {})
        return {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0)
        }