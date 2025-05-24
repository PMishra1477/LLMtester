
import os
import argparse
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import logging
import traceback

from test_executor import TestExecutor
from client_factory import ClientFactory
from evaluators.evaluator_factory import EvaluatorFactory
from utils.logger import get_logger, setup_logging
from utils.file_utils import ensure_directory_exists, load_yaml, save_yaml

logger = get_logger(__name__)

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Foundation Model Testing Framework")

    parser.add_argument(
        "--config",
        default="configs/test_config.yaml",
        help="Path to test configuration file"
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to store test results"
    )

    parser.add_argument(
        "--models",
        nargs="*",
        help="List of specific models to test (leave empty to use config file)"
    )

    parser.add_argument(
        "--categories",
        nargs="*",
        help="List of test categories to run (leave empty to use all)"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available test cases and exit"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()

def list_available_models():
    """
    List all available models from configuration.
    """
    factory = ClientFactory()
    # Change from get_available_models to get_model_names if that's the correct method
    models = factory.get_available_models()  # This might need to be changed based on the actual method name

    print("\nAvailable Models:")
    print("====")

    # Group by provider
    models_by_provider = {}
    for model in models:
        config = factory.model_configs.get(model, {})
        provider = config.get("provider", "unknown")

        if provider not in models_by_provider:
            models_by_provider[provider] = []

        models_by_provider[provider].append(model)

    # Print models by provider
    for provider, provider_models in models_by_provider.items():
        print(f"\n{provider.upper()}:")
        for model in sorted(provider_models):
            print(f"  - {model}")

    print(f"\nTotal: {len(models)} models\n")

def list_available_tests():
    """
    List all available test cases.
    """
    test_cases_dir = "test_cases"
    if not os.path.exists(test_cases_dir):
        print(f"Test cases directory not found: {test_cases_dir}")
        return

    print("\nAvailable Test Cases:")
    print("=====================")

    categories = [d for d in os.listdir(test_cases_dir)
                 if os.path.isdir(os.path.join(test_cases_dir, d))]

    total_tests = 0
    for category in sorted(categories):
        category_dir = os.path.join(test_cases_dir, category)
        test_files = [f for f in os.listdir(category_dir)
                     if f.endswith('.json')]

        print(f"\n{category.upper()} ({len(test_files)} tests):")
        for test_file in sorted(test_files):
            test_name = test_file[:-5]  # Remove .json extension
            print(f"  - {test_name}")

        total_tests += len(test_files)

    print(f"\nTotal: {total_tests} test cases across {len(categories)} categories\n")

def run_test_suite(args):
    """
    Run the complete test suite based on arguments.

    Args:
        args: Parsed command line arguments
    """
    # Initialize test executor
    executor = TestExecutor(
        test_config_path=args.config,
        output_dir=args.output_dir,
        log_level=args.log_level
    )

    # Override configuration if specified in arguments
    if args.models:
        executor.test_config["selected_models"] = args.models
        logger.info(f"Overriding models from command line: {args.models}")

    if args.categories:
        executor.test_config["categories"] = args.categories
        logger.info(f"Overriding categories from command line: {args.categories}")

    # Load test cases
    executor.load_test_cases()

    # Initialize model clients
    executor.initialize_clients()

    # Execute tests
    executor.execute_tests()

    # Generate report
    report_path = executor.generate_report()

    logger.info("Test suite execution completed successfully")
    logger.info(f"Report available at: {report_path}")

    return report_path

load_dotenv()
def main():
    """
    Main entry point for the testing framework.
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(log_level=getattr(logging, args.log_level))

    # Handle informational commands
    if args.list_models:
        list_available_models()
        sys.exit(0)

    if args.list_tests:
        list_available_tests()
        sys.exit(0)

    # Run the test suite
    try:
        report_path = run_test_suite(args)
        print(f"\nTesting completed successfully!")
        print(f"Report available at: {report_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        logger.error(traceback.format_exc())
        print(f"\nTesting failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()