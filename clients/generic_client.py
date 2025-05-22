# clients/generic_client.py
import time
import requests
from typing import Dict, Any, Tuple, Optional
from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class GenericClient(BaseClient):
    """Generic client for other LLM providers"""

    def __init__(self, model: str, config: Dict[str, Any]):
        super().__init__(model, config)
        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError(f"API URL must be provided for generic client with model {model}")

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with a generic provider's API"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Prepare request data based on provider format
            request_format = self.config.get("request_format", "openai")

            if request_format == "openai":
                request_data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": params.get("max_tokens", 1000),
                    "temperature": params.get("temperature", 0.7)
                }
            else:
                # Custom format specified in config
                request_template = self.config.get("request_template", {})
                request_data = request_template.copy()

                # Insert prompt into the appropriate location
                prompt_path = self.config.get("prompt_path", ["prompt"])
                current = request_data
                for i, key in enumerate(prompt_path):
                    if i == len(prompt_path) - 1:
                        current[key] = prompt
                    else:
                        if key not in current:
                            current[key] = {}
                        current = current[key]

                # Insert parameters
                for param_key, config_path in self.config.get("param_mapping", {}).items():
                    if param_key in params:
                        current = request_data
                        for i, key in enumerate(config_path):
                            if i == len(config_path) - 1:
                                current[key] = params[param_key]
                            else:
                                if key not in current:
                                    current[key] = {}
                                current = current[key]

            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Add any custom headers
            for key, value in self.config.get("headers", {}).items():
                headers[key] = value

            response = requests.post(
                self.api_url,
                json=request_data,
                headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            # End timing
            elapsed_time = time.time() - start_time

            # Extract content based on provider's response format
            content_path = self.config.get("content_path", ["choices", 0, "message", "content"])
            content = response_data
            for key in content_path:
                if isinstance(key, int):
                    content = content[key]
                else:
                    content = content.get(key, "")

            # Extract usage metrics
            usage = self._extract_usage(response_data)
            usage["elapsed_time"] = elapsed_time

            result = {
                "success": True,
                "content": content,
                "model": self.model,
                "provider": self.config.get("provider", "other")
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from generic model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            # Prepare request data based on provider format
            request_format = self.config.get("request_format", "openai")

            if request_format == "openai":
                request_data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": params.get("max_tokens", 1000),
                    "temperature": params.get("temperature", 0.7)
                }
            else:
                # Custom format specified in config
                request_template = self.config.get("request_template", {})
                request_data = request_template.copy()

                # Insert prompt into the appropriate location
                prompt_path = self.config.get("prompt_path", ["prompt"])
                current = request_data
                for i, key in enumerate(prompt_path):
                    if i == len(prompt_path) - 1:
                        current[key] = full_prompt
                    else:
                        if key not in current:
                            current[key] = {}
                        current = current[key]

            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Add any custom headers
            for key, value in self.config.get("headers", {}).items():
                headers[key] = value

            response = requests.post(
                self.api_url,
                json=request_data,
                headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            # Extract content based on provider's response format
            content_path = self.config.get("content_path", ["choices", 0, "message", "content"])
            content = response_data
            for key in content_path:
                if isinstance(key, int):
                    content = content[key]
                else:
                    content = content.get(key, "")

            return content
        except Exception as e:
            logger.error(f"Error generating text with {self.model}: {e}")
            raise

    def _extract_usage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token usage from API response"""
        # Try to extract usage based on configuration
        usage_path = self.config.get("usage_path", ["usage"])

        try:
            usage_data = response_data
            for key in usage_path:
                usage_data = usage_data.get(key, {})

            # Map fields according to configuration
            field_mapping = self.config.get("usage_field_mapping", {
                "input_tokens": "prompt_tokens",
                "output_tokens": "completion_tokens",
                "total_tokens": "total_tokens"
            })

            usage = {}
            for target_field, source_field in field_mapping.items():
                usage[target_field] = usage_data.get(source_field, 0)

            return usage
        except:
            # Fallback to estimation
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated": True
            }