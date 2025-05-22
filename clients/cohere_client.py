# clients/cohere_client.py
import os
import time
import cohere
import requests
from typing import Dict, Any, Optional, Tuple

from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class CohereClient(BaseClient):
    """Client implementation for Cohere API with v2 chat endpoint support."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_key = os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")

        # Check if custom API URL is provided (for v2 chat endpoint)
        self.custom_api_url = config.get("api_url")
        self.model_id = config.get("version", model_name)  # Use version field for actual model ID
        
        if self.custom_api_url:
            logger.info(f"Using custom endpoint for {model_name}: {self.custom_api_url}")
            self.use_custom_endpoint = True
        else:
            logger.info(f"Using Cohere SDK for {model_name}")
            self.use_custom_endpoint = False
            self.client = cohere.Client(api_key=self.api_key)

        logger.info(f"Initialized Cohere client for model: {model_name} (ID: {self.model_id})")

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with Cohere models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            if self.use_custom_endpoint:
                response_data = self._call_custom_chat_endpoint(prompt, params)
                content = response_data.get("message", {}).get("content", [{}])[0].get("text", "")
            else:
                # Use Cohere SDK (v1 generate endpoint)
                response = self.client.generate(
                    prompt=prompt,
                    model=self.model_id,
                    max_tokens=params.get("max_tokens", 1024),
                    temperature=params.get("temperature", 0.7),
                    **params.get("parameters", {})
                )
                content = response.generations[0].text

            # End timing
            elapsed_time = time.time() - start_time

            # Extract usage metrics
            usage = self._extract_usage_custom(response_data) if self.use_custom_endpoint else self._extract_usage(response)
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

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from the Cohere model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            if self.use_custom_endpoint:
                response_data = self._call_custom_chat_endpoint(full_prompt, params)
                return response_data.get("message", {}).get("content", [{}])[0].get("text", "")
            else:
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

    def _call_custom_chat_endpoint(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Cohere v2 chat endpoint"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Cohere v2 chat API format
        request_data = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": params.get("max_tokens", 1024),
            "temperature": params.get("temperature", 0.7),
            "p": params.get("top_p", 0.9)
        }

        response = requests.post(
            self.custom_api_url,
            json=request_data,
            headers=headers,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        """Extract token usage from Cohere SDK response"""
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

    def _extract_usage_custom(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token usage from Cohere v2 chat response"""
        try:
            usage = response_data.get("usage", {})
            return {
                "input_tokens": usage.get("tokens", {}).get("input_tokens", 0),
                "output_tokens": usage.get("tokens", {}).get("output_tokens", 0),
                "total_tokens": usage.get("tokens", {}).get("input_tokens", 0) + usage.get("tokens", {}).get("output_tokens", 0)
            }
        except Exception:
            # Fallback to estimation
            content = response_data.get("message", {}).get("content", [{}])[0].get("text", "")
            output_tokens = len(content) // 4
            return {
                "input_tokens": 0,
                "output_tokens": output_tokens,
                "total_tokens": output_tokens,
                "estimated": True
            }

    def count_tokens(self, text: str) -> int:
        """Count tokens in the input text."""
        if not self.use_custom_endpoint:
            try:
                tokens_response = self.client.tokenize(text)
                return len(tokens_response.tokens)
            except Exception:
                pass
        
        # Fallback method
        return len(text) // 4