
import os
import importlib
from typing import Dict, Any, Optional, List, Type
import traceback

from clients.base_client import BaseClient
from clients.openai_client import OpenAIClient
from clients.anthropic_client import AnthropicClient
from clients.google_client import GoogleClient
from clients.meta_client import MetaClient
from clients.mistral_client import MistralClient
from clients.generic_client import GenericClient
from clients.cohere_client import CohereClient
from utils.logger import get_logger
from utils.file_utils import load_yaml, list_files

logger = get_logger(__name__)

class ClientFactory:
    """
    Factory class for creating model clients based on configuration.
    Manages client creation for different provider types.
    """

    # Map of provider names to client classes
    _CLIENT_CLASSES = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "meta": MetaClient,
        "mistral": MistralClient,
        "cohere": CohereClient,
        "generic": GenericClient
    }

    def __init__(self, config_dir: str = "configs/models"):
        """
        Initialize the client factory.

        Args:
            config_dir: Directory containing model configuration files
        """
        self.config_dir = config_dir
        self.model_configs = self._load_model_configs()
        logger.info(f"Loaded {len(self.model_configs)} model configurations")

    def initialize_clients(self):
        """
        Initialize model clients based on configuration.
        """
        factory = ClientFactory()
        selected_models = self.test_config.get("selected_models", [])

        if not selected_models:
            # If no models specified, use all available
            selected_models = factory.get_available_models()

        logger.info(f"Initializing clients for models: {selected_models}")

        for model_name in selected_models:
            try:
                # Change from get_client to create_client as suggested by the error
                client = factory.create_client(model_name)
                self.clients[model_name] = client
                logger.debug(f"Initialized client for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize client for model {model_name}: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Initialized {len(self.clients)} model clients")

    def _load_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all model configurations from the config directory.

        Returns:
            Dictionary mapping model names to their configurations
        """
        configs = {}

        # Ensure config directory exists
        if not os.path.exists(self.config_dir):
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return configs

        # Load all YAML files in the config directory
        yaml_files = list_files(self.config_dir, extension=".yaml")
        for file_path in yaml_files:
            try:
                config = load_yaml(file_path)
                if isinstance(config, dict) and "models" in config:
                    for model_name, model_config in config["models"].items():
                        # Add provider info from file if not explicitly set
                        if "provider" not in model_config:
                            # Extract provider from filename (e.g., openai.yaml -> openai)
                            provider = os.path.basename(file_path).split(".")[0]
                            model_config["provider"] = provider
                        
                        configs[model_name] = model_config
                        logger.debug(f"Loaded configuration for model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading config from {file_path}: {e}")

        return configs

    def get_available_models(self) -> List[str]:
        """
        Get list of all available configured models.

        Returns:
            List of model names
        """
        return list(self.model_configs.keys())

    def get_models_by_provider(self, provider: str) -> List[str]:
        """
        Get list of models for a specific provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            List of model names for the provider
        """
        return [
            model_name for model_name, config in self.model_configs.items()
            if config.get("provider", "").lower() == provider.lower()
        ]

    def create_client(self, model_name: str) -> BaseClient:
        """
        Create a client for the specified model.

        Args:
            model_name: Name of the model to create a client for

        Returns:
            Appropriate client instance for the model

        Raises:
            ValueError: If model is not found or client creation fails
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in configurations")

        model_config = self.model_configs[model_name]
        provider = model_config.get("provider", "generic").lower()

        # Get the appropriate client class
        client_class = self._CLIENT_CLASSES.get(provider)
        if not client_class:
            logger.warning(f"No specific client for provider '{provider}', using generic client")
            client_class = GenericClient

        try:
            # Create and return client instance
            client = client_class(model_name, model_config)
            logger.info(f"Created {provider} client for model: {model_name}")
            return client
        except Exception as e:
            logger.error(f"Error creating client for model {model_name}: {e}")
            raise ValueError(f"Failed to create client for model {model_name}: {e}")
    
    def get_client(self, model_name: str) -> BaseClient:
        """
        Create a client for the specified model.

        Args:
            model_name: Name of the model to create a client for

        Returns:
            Appropriate client instance for the model

        Raises:
            ValueError: If model is not found or client creation fails
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in configurations")

        model_config = self.model_configs[model_name]
        provider = model_config.get("provider", "generic").lower()

        # Get the appropriate client class
        client_class = self._CLIENT_CLASSES.get(provider)
        if not client_class:
            logger.warning(f"No specific client for provider '{provider}', using generic client")
            client_class = GenericClient

        try:
            # Create and return client instance
            client = client_class(model_name, model_config)
            logger.info(f"Created {provider} client for model: {model_name}")
            return client
        except Exception as e:
            logger.error(f"Error creating client for model {model_name}: {e}")
            raise ValueError(f"Failed to create client for model {model_name}: {e}")

    def create_all_clients(self) -> Dict[str, BaseClient]:
        """
        Create clients for all configured models.

        Returns:
            Dictionary mapping model names to client instances
        """
        clients = {}
        for model_name in self.get_available_models():
            try:
                clients[model_name] = self.create_client(model_name)
            except Exception as e:
                logger.error(f"Skipping model {model_name} due to error: {e}")
        
        return clients