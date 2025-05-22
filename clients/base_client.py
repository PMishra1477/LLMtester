# clients/base_client.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class BaseClient(ABC):
    """Abstract base class for model API clients"""

    def __init__(self, model: str, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.api_key = self._get_api_key()
        self.initialized = self._initialize()

    @abstractmethod
    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a test case and return the response and usage data

        Args:
            test_case: Test case definition with prompt and parameters

        Returns:
            Tuple containing:
                - Response data (including content and metadata)
                - Usage data (tokens, costs, etc.)
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text from the model - compatibility method for TestExecutor.
        
        Args:
            prompt: The main prompt text
            context: Optional context to prepend to prompt
            parameters: Optional parameters for generation
            
        Returns:
            Generated text response
        """
        pass

    def _get_api_key(self) -> str:
        """Get API key from config or environment variables"""
        if "api_key" in self.config:
            return self.config["api_key"]

        import os
        provider = self.config.get("provider", "unknown")
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)

        if not api_key:
            raise ValueError(f"API key not found in config or environment variable {env_var_name}")

        return api_key

    def _initialize(self) -> bool:
        """Initialize the client (setup connections, validate credentials)"""
        return True

    def _prepare_prompt(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the prompt and parameters for API call"""
        prompt = test_case.get("prompt", "")

        # Get parameters from test case or use defaults
        params = test_case.get("parameters", {}).copy()

        # Apply model-specific parameter defaults
        default_params = self.config.get("parameters", {})
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

        return {
            "prompt": prompt,
            "parameters": params
        }

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        """Extract usage information from the response"""
        # Default implementation (subclasses should override)
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    def _handle_error(self, error: Exception, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Handle API errors gracefully"""
        error_response = {
            "success": False,
            "error": str(error),
            "content": f"Error executing {self.model}: {str(error)}"
        }

        # Empty usage for errors
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": str(error)
        }

        return error_response, usage