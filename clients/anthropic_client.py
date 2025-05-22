# clients/anthropic_client.py
import time
from typing import Dict, Any, Tuple, Optional
import anthropic
from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class AnthropicClient(BaseClient):
    """Client for Anthropic models (Claude series)"""

    def __init__(self, model: str, config: Dict[str, Any]):
        super().__init__(model, config)
        self.client = anthropic.Anthropic(api_key=self.api_key)
        # Use version field if available, otherwise use model name
        self.model_id = config.get("version", model)
        logger.info(f"Initialized Anthropic client for {model} (API model: {self.model_id})")

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with Anthropic models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Make API call using the actual model ID
            response = self.client.messages.create(
                model=self.model_id,  # Use the version/model_id
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                system=params.get("system_prompt", "You are a helpful AI assistant."),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # End timing
            elapsed_time = time.time() - start_time

            # Process response
            content = response.content[0].text

            # Extract usage metrics
            usage = self._extract_usage(response)
            usage["elapsed_time"] = elapsed_time

            result = {
                "success": True,
                "content": content,
                "model": self.model,  # Keep original model name for tracking
                "api_model": self.model_id,  # Add actual API model ID
                "provider": "anthropic"
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from Anthropic model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            response = self.client.messages.create(
                model=self.model_id,  # Use the version/model_id
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                system=params.get("system_prompt", "You are a helpful AI assistant."),
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with {self.model} (API: {self.model_id}): {e}")
            raise

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        """Extract token usage from Anthropic response"""
        # Anthropic includes usage info in response
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        return usage