# Foundation Model Testing Framework

A comprehensive testing framework for evaluating and comparing AI-powered presentation generation across different foundation language models.

## Overview

This framework provides automated, structured testing for foundation models with a focus on presentation generation capabilities. It allows for:

- Testing multiple models from different providers (OpenAI, Anthropic, Google, Meta, Mistral, etc.)
- Evaluating performance across various dimensions (content, formatting, consistency, creativity, etc.)
- Generating comprehensive comparison reports
- Tracking API usage and costs

## Directory Structure

foundation-model-testing/
├── clients/ # Model client implementations
│ ├── base_client.py # Base interface for model clients
│ ├── openai_client.py # OpenAI-specific implementation
│ ├── anthropic_client.py # Anthropic-specific implementation
│ └── ... # Other model clients
├── configs/ # Configuration files
│ ├── models/ # Model-specific configurations
│ │ ├── openai.yaml # OpenAI models configuration
│ │ ├── anthropic.yaml # Anthropic models configuration
│ │ └── ... # Other model configurations
│ └── test_config.yaml # Test execution configuration
├── evaluators/ # Test evaluation logic
│ ├── base_evaluator.py # Base evaluation interface
│ ├── content_evaluator.py # Content quality evaluation
│ ├── ppt_evaluator.py # Presentation-specific evaluation
│ └── ... # Other specialized evaluators
├── reporting/ # Reporting utilities
│ ├── comparison_report.py # Model comparison reporting
│ └── visualizations.py # Data visualization tools
├── test_cases/ # Test case definitions
│ ├── content/ # Content quality test cases
│ ├── formatting/ # Formatting test cases
│ ├── consistency/ # Consistency test cases
│ └── ... # Other test categories
├── utils/ # Utility functions
│ ├── logger.py # Logging utilities
│ ├── file_utils.py # File operations utilities
│ └── cost_tracker.py # API cost tracking
├── client_factory.py # Factory for creating model clients
├── test_executor.py # Core test execution engine
├── test_integration.py # Integration and CLI entry point
└── README.md # Framework documentation


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/foundation-model-testing.git
   cd foundation-model-testing
Install dependencies:
bash
Copy Code
pip install -r requirements.txt
Configure API keys: Create a .env file in the root directory with your API keys:
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
# Add other API keys as needed
Configuration
Model Configuration
Model configurations are stored in YAML files in the configs/models/ directory. Each file contains configurations for models from a specific provider.

Example openai.yaml:

yaml
Copy Code
models:
  gpt-4o:
    model_id: gpt-4o
    max_tokens: 4096
    temperature: 0.7
    parameters:
      top_p: 1.0
  
  gpt-3.5-turbo:
    model_id: gpt-3.5-turbo
    max_tokens: 4096
    temperature: 0.7
    parameters:
      top_p: 1.0
Test Configuration
The main test configuration is defined in configs/test_config.yaml:

yaml
Copy Code
# Test configuration
model_config_dir: "configs/models"
test_cases_dir: "test_cases"

# Models to test (empty list means all available models)
selected_models:
  - gpt-4o
  - claude-3-opus
  - gemini-1.5-pro
  - mistral-large
  - llama-3-70b

# Test categories to include (empty list means all categories)
categories:
  - content
  - formatting
  - consistency
  - creativity
  - efficiency
  - personalization
  - ppt

# Metric configuration
metrics:
  primary: "overall_score"
  weights:
    content: 0.25
    formatting: 0.15
    consistency: 0.15
    creativity: 0.15
    efficiency: 0.15
    personalization: 0.10
    ppt: 0.05

# Test execution settings
parallel_execution: false
retry_count: 2
timeout: 300  # seconds
Usage
Basic Usage
Run the complete test suite with default settings:

bash
Copy Code
python -m foundation_model_testing
Advanced Usage
Specify models and categories to test:

bash
Copy Code
python -m foundation_model_testing --models gpt-4o claude-3-opus gemini-1.5-pro --categories content formatting
List available models and test cases:

bash
Copy Code
python -m foundation_model_testing --list-models
python -m foundation_model_testing --list-tests
Specify custom configuration and output directory:

bash
Copy Code
python -m foundation_model_testing --config my_config.yaml --output-dir my_results
Reports
After test execution, comprehensive reports are generated in the output directory (default: results/reports/). These include:

Overall performance comparison
Category-specific performance analysis
Cost analysis
Visualizations (charts, heatmaps, etc.)
Extending the Framework
Adding New Models
Create a configuration file in configs/models/ for the new provider if needed
Add model details to the configuration file
Implement a client class in clients/ if the provider isn't already supported
Register the client in client_factory.py
Adding New Test Cases
Create a JSON test case file in the appropriate category directory under test_cases/
Follow the test case schema with prompts, expected elements, and evaluation criteria
Adding New Evaluators
Create a new evaluator class in evaluators/ implementing the BaseEvaluator interface
Add logic specific to your evaluation needs
Register the evaluator in evaluator_factory.py
License
MIT License