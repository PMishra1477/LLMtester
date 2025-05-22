# clients/cohere_client.py
import os
import time
import cohere
from typing import Dict, Any, Optional, Tuple

from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class CohereClient(BaseClient):
    """Client implementation for Cohere API."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_key = os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")

        self.client = cohere.Client(api_key=self.api_key)
        self.model_id = config.get("model_id", model_name)
        logger.info(f"Initialized Cohere client for model: {model_name}")

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with Cohere models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Make API call
            response = self.client.generate(
                prompt=prompt,
                model=self.model_id,
                max_tokens=params.get("max_tokens", 1024),
                temperature=params.get("temperature", 0.7),
                **params.get("parameters", {})
            )

            # End timing
            elapsed_time = time.time() - start_time

            # Process response
            content = response.generations[0].text

            # Extract usage metrics
            usage = self._extract_usage(response)
            usage["elapsed_time"] = elapsed_time

            result = {
                "success": True,
                "content": content,
                "model": self.model,
                "provider": "cohere"
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        """Extract token usage from Cohere response"""
        # Cohere doesn't provide detailed token usage in the same way
        # We'll estimate based on the response
        try:
            input_tokens = len(response.prompt) // 4  # Rough estimate
            output_tokens = len(response.generations[0].text) // 4  # Rough estimate
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        except Exception:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated": True
            }

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from the Cohere model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            response = self.client.generate(
                prompt=full_prompt,
                model=self.model_id,
                max_tokens=params.get("max_tokens", 1024),
                temperature=params.get("temperature", 0.7),
                **params
            )
            return response.generations[0].text
        except Exception as e:
            logger.error(f"Error generating text with {self.model}: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in the input text."""
        try:
            tokens_response = self.client.tokenize(text)
            return len(tokens_response.tokens)
        except Exception:
            # Fallback method if tokenize API call fails
            # Rough estimate: 4 characters per token
            return len(text) // 4