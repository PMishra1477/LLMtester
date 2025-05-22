# clients/openai_client.py
import time
from typing import Dict, Any, Tuple, Optional
import openai
from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIClient(BaseClient):
    """Client for OpenAI models (GPT-3.5, GPT-4, etc.)"""

    def __init__(self, model: str, config: Dict[str, Any]):
        super().__init__(model, config)
        self.client = openai.OpenAI(api_key=self.api_key)

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with OpenAI models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                frequency_penalty=params.get("frequency_penalty", 0.0),
                presence_penalty=params.get("presence_penalty", 0.0)
            )

            # End timing
            elapsed_time = time.time() - start_time

            # Process response
            content = response.choices[0].message.content

            # Extract usage metrics
            usage = self._extract_usage(response)
            usage["elapsed_time"] = elapsed_time

            result = {
                "success": True,
                "content": content,
                "model": self.model,
                "provider": "openai"
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from OpenAI model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                frequency_penalty=params.get("frequency_penalty", 0.0),
                presence_penalty=params.get("presence_penalty", 0.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with {self.model}: {e}")
            raise

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        """Extract token usage from OpenAI response"""
        usage = response.usage
        return {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }