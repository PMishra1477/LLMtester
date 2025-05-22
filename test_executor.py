import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import traceback

from client_factory import ClientFactory
from evaluators.evaluator_factory import EvaluatorFactory
from utils.logger import get_logger
from utils.file_utils import ensure_directory_exists, load_yaml, save_yaml, save_json

logger = get_logger(__name__)

class TestExecutor:
    """
    Main class responsible for executing tests across multiple models and test cases.
    """

    def __init__(self, test_config_path: str, output_dir: str, log_level: str = "INFO"):
        """
        Initialize the test executor.

        Args:
            test_config_path: Path to the test configuration file
            output_dir: Directory to store test results
            log_level: Logging level
        """
        self.test_config_path = test_config_path
        self.output_dir = output_dir
        self.log_level = log_level

        # Load test configuration
        self.test_config = load_yaml(test_config_path)
        if not self.test_config:
            raise ValueError(f"Failed to load test configuration from {test_config_path}")

        # Initialize empty containers
        self.test_cases = {}
        self.clients = {}
        self.results = {}

        # Create output directory
        ensure_directory_exists(output_dir)

        # Create run-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        ensure_directory_exists(self.run_dir)

        # Save a copy of the configuration
        # Fix: Swap the parameter order - file path first, then data
        save_yaml(os.path.join(self.run_dir, "config.yaml"), self.test_config)

        logger.info(f"Test executor initialized with config: {test_config_path}")
        logger.info(f"Results will be stored in: {self.run_dir}")

    def load_test_cases(self):
        """
        Load test cases from the test_cases directory based on configuration.
        """
        test_cases_dir = "test_cases"
        categories = self.test_config.get("categories", [])

        if not categories:
            # If no categories specified, use all available
            categories = [d for d in os.listdir(test_cases_dir)
                         if os.path.isdir(os.path.join(test_cases_dir, d))]

        logger.info(f"Loading test cases for categories: {categories}")

        for category in categories:
            category_dir = os.path.join(test_cases_dir, category)
            if not os.path.exists(category_dir):
                logger.warning(f"Category directory not found: {category_dir}")
                continue

            self.test_cases[category] = {}
            test_files = [f for f in os.listdir(category_dir) if f.endswith('.json')]

            for test_file in test_files:
                test_path = os.path.join(category_dir, test_file)
                test_name = test_file[:-5]  # Remove .json extension

                try:
                    with open(test_path, 'r') as f:
                        test_case = json.load(f)
                    self.test_cases[category][test_name] = test_case
                    logger.debug(f"Loaded test case: {category}/{test_name}")
                except Exception as e:
                    logger.error(f"Error loading test case {test_path}: {e}")

        total_tests = sum(len(tests) for tests in self.test_cases.values())
        logger.info(f"Loaded {total_tests} test cases across {len(self.test_cases)} categories")

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
                client = factory.get_client(model_name)
                self.clients[model_name] = client
                logger.debug(f"Initialized client for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize client for model {model_name}: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Initialized {len(self.clients)} model clients")

    def execute_tests(self):
        """
        Execute all test cases across all models.
        """
        if not self.test_cases:
            raise ValueError("No test cases loaded. Call load_test_cases() first.")

        if not self.clients:
            raise ValueError("No model clients initialized. Call initialize_clients() first.")

        # Initialize results structure
        for model_name in self.clients.keys():
            self.results[model_name] = {}
            for category in self.test_cases.keys():
                self.results[model_name][category] = {}

        # Create evaluator factory
        evaluator_factory = EvaluatorFactory()

        # Track execution statistics
        total_tests = sum(len(tests) for tests in self.test_cases.values())
        total_models = len(self.clients)
        total_executions = total_tests * total_models
        completed = 0
        start_time = time.time()

        logger.info(f"Starting test execution: {total_executions} total executions "
                   f"({total_tests} tests × {total_models} models)")

        # Execute tests for each model and category
        for category, tests in self.test_cases.items():
            # Get the appropriate evaluator for this category
            evaluator = evaluator_factory.get_evaluator(category)
            if not evaluator:
                logger.warning(f"No evaluator found for category: {category}, skipping")
                continue

            category_dir = os.path.join(self.run_dir, category)
            ensure_directory_exists(category_dir)

            for test_name, test_case in tests.items():
                test_dir = os.path.join(category_dir, test_name)
                ensure_directory_exists(test_dir)

                # Save the test case for reference
                # Fix: Swap parameter order - file path first, then data
                save_json(os.path.join(test_dir, "test_case.json"), test_case)

                for model_name, client in self.clients.items():
                    model_dir = os.path.join(test_dir, model_name)
                    ensure_directory_exists(model_dir)

                    logger.info(f"Executing test: {category}/{test_name} with model: {model_name}")

                    try:
                        # Execute the test
                        prompt = test_case.get("prompt", "")
                        context = test_case.get("context", "")
                        parameters = test_case.get("parameters", {})

                        # Record start time
                        test_start_time = time.time()

                        # Get response from model
                        response = client.generate(prompt, context, parameters)

                        # Record end time and calculate duration
                        test_end_time = time.time()
                        duration = test_end_time - test_start_time

                        # Evaluate the response
                        evaluation = evaluator.evaluate(test_case, response)

                        # Add metadata
                        result = {
                            "test_case": test_case,
                            "response": response,
                            "evaluation": evaluation,
                            "metadata": {
                                "model": model_name,
                                "category": category,
                                "test_name": test_name,
                                "timestamp": datetime.now().isoformat(),
                                "duration": duration
                            }
                        }

                        # Save result
                        # Fix: Swap parameter order - file path first, then data
                        save_json(os.path.join(model_dir, "result.json"), result)

                        # Store in results dictionary
                        self.results[model_name][category][test_name] = result

                        logger.debug(f"Test completed: {category}/{test_name} with model: {model_name}")
                        logger.debug(f"Score: {evaluation.get('score', 'N/A')}, Duration: {duration:.2f}s")

                    except Exception as e:
                        logger.error(f"Error executing test {category}/{test_name} with model {model_name}: {e}")
                        logger.error(traceback.format_exc())

                        # Record failure
                        error_result = {
                            "test_case": test_case,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "metadata": {
                                "model": model_name,
                                "category": category,
                                "test_name": test_name,
                                "timestamp": datetime.now().isoformat(),
                                "status": "failed"
                            }
                        }

                        # Save error result
                        # Fix: Swap parameter order - file path first, then data
                        save_json(os.path.join(model_dir, "error.json"), error_result)

                        # Store in results dictionary
                        self.results[model_name][category][test_name] = error_result

                    # Update progress
                    completed += 1
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = avg_time * (total_executions - completed)

                    logger.info(f"Progress: {completed}/{total_executions} "
                               f"({completed/total_executions*100:.1f}%) - "
                               f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")

        # Save complete results
        # Fix: Swap parameter order - file path first, then data
        save_json(os.path.join(self.run_dir, "all_results.json"), self.results)

        total_time = time.time() - start_time
        logger.info(f"Test execution completed: {completed}/{total_executions} tests in {total_time:.1f}s")

        return self.results

    def generate_report(self):
        """
        Generate a comprehensive report from test results.

        Returns:
            Path to the generated report
        """
        if not self.results:
            raise ValueError("No test results available. Execute tests first.")

        logger.info("Generating test report...")

        # Create report directory
        report_dir = os.path.join(self.run_dir, "report")
        ensure_directory_exists(report_dir)

        # Aggregate results
        aggregated_results = self._aggregate_results()

        # Save aggregated results
        # Fix: Swap parameter order - file path first, then data
        save_json(os.path.join(report_dir, "aggregated_results.json"), aggregated_results)

        # Generate summary report
        summary = self._generate_summary(aggregated_results)
        # Fix: Swap parameter order - file path first, then data
        save_json(os.path.join(report_dir, "summary.json"), summary)

        # Generate HTML report
        html_report_path = os.path.join(report_dir, "report.html")
        self._generate_html_report(summary, aggregated_results, html_report_path)

        logger.info(f"Report generated at: {report_dir}")

        return report_dir

    def _aggregate_results(self):
        """
        Aggregate test results for reporting.

        Returns:
            Dictionary with aggregated results
        """
        aggregated = {
            "models": {},
            "categories": {},
            "overall": {
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "average_score": 0,
                "average_duration": 0
            }
        }

        total_score = 0
        total_duration = 0
        test_count = 0

        # Aggregate by model and category
        for model_name, model_results in self.results.items():
            if model_name not in aggregated["models"]:
                aggregated["models"][model_name] = {
                    "total_tests": 0,
                    "passed": 0,
                    "failed": 0,
                    "average_score": 0,
                    "average_duration": 0,
                    "categories": {}
                }

            model_total_score = 0
            model_total_duration = 0
            model_test_count = 0

            for category, category_results in model_results.items():
                if category not in aggregated["categories"]:
                    aggregated["categories"][category] = {
                        "total_tests": 0,
                        "passed": 0,
                        "failed": 0,
                        "average_score": 0,
                        "average_duration": 0,
                        "models": {}
                    }

                if category not in aggregated["models"][model_name]["categories"]:
                    aggregated["models"][model_name]["categories"][category] = {
                        "total_tests": 0,
                        "passed": 0,
                        "failed": 0,
                        "average_score": 0,
                        "average_duration": 0
                    }

                if model_name not in aggregated["categories"][category]["models"]:
                    aggregated["categories"][category]["models"][model_name] = {
                        "total_tests": 0,
                        "passed": 0,
                        "failed": 0,
                        "average_score": 0,
                        "average_duration": 0
                    }

                category_total_score = 0
                category_total_duration = 0
                category_test_count = 0

                for test_name, result in category_results.items():
                    # Check if test failed
                    failed = "error" in result

                    # Get score and duration
                    score = 0
                    duration = 0

                    if not failed:
                        evaluation = result.get("evaluation", {})
                        score = evaluation.get("score", 0)
                        metadata = result.get("metadata", {})
                        duration = metadata.get("duration", 0)

                    # Update counts
                    aggregated["models"][model_name]["total_tests"] += 1
                    aggregated["categories"][category]["total_tests"] += 1
                    aggregated["models"][model_name]["categories"][category]["total_tests"] += 1
                    aggregated["categories"][category]["models"][model_name]["total_tests"] += 1
                    aggregated["overall"]["total_tests"] += 1

                    if failed:
                        aggregated["models"][model_name]["failed"] += 1
                        aggregated["categories"][category]["failed"] += 1
                        aggregated["models"][model_name]["categories"][category]["failed"] += 1
                        aggregated["categories"][category]["models"][model_name]["failed"] += 1
                        aggregated["overall"]["total_failed"] += 1
                    else:
                        aggregated["models"][model_name]["passed"] += 1
                        aggregated["categories"][category]["passed"] += 1
                        aggregated["models"][model_name]["categories"][category]["passed"] += 1
                        aggregated["categories"][category]["models"][model_name]["passed"] += 1
                        aggregated["overall"]["total_passed"] += 1

                        # Update scores and durations
                        category_total_score += score
                        category_total_duration += duration
                        category_test_count += 1

                        model_total_score += score
                        model_total_duration += duration
                        model_test_count += 1

                        total_score += score
                        total_duration += duration
                        test_count += 1

                # Calculate averages for this model/category combination
                if category_test_count > 0:
                    avg_score = category_total_score / category_test_count
                    avg_duration = category_total_duration / category_test_count

                    aggregated["models"][model_name]["categories"][category]["average_score"] = avg_score
                    aggregated["models"][model_name]["categories"][category]["average_duration"] = avg_duration
                    aggregated["categories"][category]["models"][model_name]["average_score"] = avg_score
                    aggregated["categories"][category]["models"][model_name]["average_duration"] = avg_duration

            # Calculate averages for this model
            if model_test_count > 0:
                aggregated["models"][model_name]["average_score"] = model_total_score / model_test_count
                aggregated["models"][model_name]["average_duration"] = model_total_duration / model_test_count

        # Calculate category averages
        for category, category_data in aggregated["categories"].items():
            category_score = 0
            category_duration = 0
            category_count = 0

            for model_name, model_data in category_data["models"].items():
                if model_data["passed"] > 0:
                    category_score += model_data["average_score"] * model_data["passed"]
                    category_duration += model_data["average_duration"] * model_data["passed"]
                    category_count += model_data["passed"]

            if category_count > 0:
                aggregated["categories"][category]["average_score"] = category_score / category_count
                aggregated["categories"][category]["average_duration"] = category_duration / category_count

        # Calculate overall averages
        if test_count > 0:
            aggregated["overall"]["average_score"] = total_score / test_count
            aggregated["overall"]["average_duration"] = total_duration / test_count

        return aggregated

    def _generate_summary(self, aggregated_results):
        """
        Generate a summary of test results.

        Args:
            aggregated_results: Aggregated test results

        Returns:
            Dictionary with summary information
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": self.test_config,
            "overall": aggregated_results["overall"],
            "model_rankings": [],
            "category_rankings": [],
            "model_category_matrix": {}
        }

        # Create model rankings
        models = []
        for model_name, model_data in aggregated_results["models"].items():
            models.append({
                "name": model_name,
                "average_score": model_data["average_score"],
                "average_duration": model_data["average_duration"],
                "passed": model_data["passed"],
                "failed": model_data["failed"],
                "total": model_data["total_tests"]
            })

        # Sort by average score (descending)
        summary["model_rankings"] = sorted(models, key=lambda x: x["average_score"], reverse=True)

        # Create category rankings
        categories = []
        for category_name, category_data in aggregated_results["categories"].items():
            categories.append({
                "name": category_name,
                "average_score": category_data["average_score"],
                "average_duration": category_data["average_duration"],
                "passed": category_data["passed"],
                "failed": category_data["failed"],
                "total": category_data["total_tests"]
            })

        # Sort by name
        summary["category_rankings"] = sorted(categories, key=lambda x: x["name"])

        # Create model/category matrix
        for model_name, model_data in aggregated_results["models"].items():
            summary["model_category_matrix"][model_name] = {}

            for category_name, category_data in aggregated_results["categories"].items():
                if category_name in model_data["categories"]:
                    model_category = model_data["categories"][category_name]
                    summary["model_category_matrix"][model_name][category_name] = {
                        "average_score": model_category["average_score"],
                        "passed": model_category["passed"],
                        "failed": model_category["failed"]
                    }
                else:
                    summary["model_category_matrix"][model_name][category_name] = {
                        "average_score": 0,
                        "passed": 0,
                        "failed": 0
                    }

        return summary

    def _generate_html_report(self, summary, aggregated_results, output_path):
        """
        Generate an HTML report from the test results.

        Args:
            summary: Summary information
            aggregated_results: Aggregated test results
            output_path: Path to save the HTML report
        """
        # Simple HTML report template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Foundation Model Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .score-high { color: green; }
                .score-medium { color: orange; }
                .score-low { color: red; }
                .summary-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Foundation Model Test Report</h1>
            <div class="summary-box">
                <h2>Summary</h2>
                <p>Timestamp: {timestamp}</p>
                <p>Total Tests: {total_tests}</p>
                <p>Passed: {passed} ({pass_rate:.1f}%)</p>
                <p>Failed: {failed}</p>
                <p>Average Score: {avg_score:.2f}</p>
                <p>Average Duration: {avg_duration:.2f}s</p>
            </div>

            <h2>Model Rankings</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Average Score</th>
                    <th>Pass Rate</th>
                    <th>Average Duration</th>
                </tr>
                {model_rows}
            </table>

            <h2>Category Performance</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Average Score</th>
                    <th>Pass Rate</th>
                    <th>Best Model</th>
                </tr>
                {category_rows}
            </table>

            <h2>Model-Category Matrix</h2>
            <table>
                <tr>
                    <th>Model / Category</th>
                    {category_headers}
                </tr>
                {matrix_rows}
            </table>
        </body>
        </html>
        """

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get overall stats
        total_tests = summary["overall"]["total_tests"]
        passed = summary["overall"]["total_passed"]
        failed = summary["overall"]["total_failed"]
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        avg_score = summary["overall"]["average_score"]
        avg_duration = summary["overall"]["average_duration"]

        # Generate model ranking rows
        model_rows = ""
        for i, model in enumerate(summary["model_rankings"]):
            model_pass_rate = (model["passed"] / model["total"] * 100) if model["total"] > 0 else 0
            score_class = "score-high" if model["average_score"] >= 0.8 else "score-medium" if model["average_score"] >= 0.5 else "score-low"

            model_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{model["name"]}</td>
                <td class="{score_class}">{model["average_score"]:.2f}</td>
                <td>{model_pass_rate:.1f}% ({model["passed"]}/{model["total"]})</td>
                <td>{model["average_duration"]:.2f}s</td>
            </tr>
            """

        # Generate category rows
        category_rows = ""
        for category in summary["category_rankings"]:
            category_pass_rate = (category["passed"] / category["total"] * 100) if category["total"] > 0 else 0
            score_class = "score-high" if category["average_score"] >= 0.8 else "score-medium" if category["average_score"] >= 0.5 else "score-low"

            # Find best model for this category
            best_model = "N/A"
            best_score = 0

            for model_name, categories in summary["model_category_matrix"].items():
                if category["name"] in categories:
                    model_category_score = categories[category["name"]]["average_score"]
                    if model_category_score > best_score:
                        best_score = model_category_score
                        best_model = model_name

            category_rows += f"""
            <tr>
                <td>{category["name"]}</td>
                <td class="{score_class}">{category["average_score"]:.2f}</td>
                <td>{category_pass_rate:.1f}% ({category["passed"]}/{category["total"]})</td>
                <td>{best_model} ({best_score:.2f})</td>
            </tr>
            """

        # Generate category headers for matrix
        category_headers = ""
        for category in summary["category_rankings"]:
            category_headers += f"<th>{category['name']}</th>"

        # Generate matrix rows
        matrix_rows = ""
        for model in summary["model_rankings"]:
            model_name = model["name"]
            row = f"<tr><td>{model_name}</td>"

            for category in summary["category_rankings"]:
                category_name = category["name"]
                if category_name in summary["model_category_matrix"][model_name]:
                    cell_data = summary["model_category_matrix"][model_name][category_name]
                    score = cell_data["average_score"]
                    score_class = "score-high" if score >= 0.8 else "score-medium" if score >= 0.5 else "score-low"

                    row += f"""
                    <td class="{score_class}">
                        {score:.2f}<br>
                        ({cell_data["passed"]}/{cell_data["passed"] + cell_data["failed"]})
                    </td>
                    """
                else:
                    row += "<td>N/A</td>"

            row += "</tr>"
            matrix_rows += row

        # Fill in the template
        html_content = html_template.format(
            timestamp=timestamp,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            avg_score=avg_score,
            avg_duration=avg_duration,
            model_rows=model_rows,
            category_rows=category_rows,
            category_headers=category_headers,
            matrix_rows=matrix_rows
        )

        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)

        return output_path