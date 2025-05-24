
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from utils.logger import get_logger
from utils.file_utils import ensure_directory_exists

logger = get_logger(__name__)

class ComparisonReport:
    """
    Generates comprehensive comparison reports between different models
    based on test results.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the comparison report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        ensure_directory_exists(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_report(self, results: Dict[str, Dict[str, Any]], 
                        test_config: Dict[str, Any],
                        cost_data: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a comprehensive comparison report from test results.
        
        Args:
            results: Dictionary of test results by model
            test_config: Test configuration used
            cost_data: Optional cost data for each model
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating comparison report")
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, f"report_{self.timestamp}")
        ensure_directory_exists(report_dir)
        
        # Convert results to DataFrame for easier manipulation
        df = self._results_to_dataframe(results)
        
        # Save raw results as JSON
        raw_results_path = os.path.join(report_dir, "raw_results.json")
        with open(raw_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary_path = self._generate_summary(df, report_dir, cost_data)
        
        # Generate detailed report with visualizations
        detailed_path = self._generate_detailed_report(df, results, test_config, report_dir, cost_data)
        
        logger.info(f"Report generated at {report_dir}")
        return report_dir
    
    def _results_to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert nested results dictionary to a flat DataFrame"""
        rows = []
        
        for model_name, model_results in results.items():
            for test_case, metrics in model_results.items():
                if isinstance(metrics, dict):
                    row = {
                        'model': model_name,
                        'test_case': test_case
                    }
                    # Flatten metrics
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            row[metric_name] = value
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_summary(self, df: pd.DataFrame, report_dir: str, 
                          cost_data: Optional[Dict[str, float]] = None) -> str:
        """Generate summary report with key metrics"""
        from visualizations import create_performance_chart, create_cost_efficiency_chart
        
        summary_path = os.path.join(report_dir, "summary.md")
        
        with open(summary_path, 'w') as f:
            f.write(f"# Model Comparison Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance table
            f.write("## Overall Performance\n\n")
            
            # Calculate average scores by model
            avg_scores = df.groupby('model').mean(numeric_only=True).round(2)
            f.write(avg_scores.to_markdown() + "\n\n")
            
            # Add cost information if available
            if cost_data:
                f.write("## Cost Analysis\n\n")
                cost_df = pd.DataFrame({
                    'model': list(cost_data.keys()),
                    'total_cost': list(cost_data.values())
                })
                f.write(cost_df.to_markdown() + "\n\n")
            
            # Add links to detailed reports
            f.write("## Detailed Reports\n\n")
            f.write("- [Detailed Metrics](./detailed_metrics.md)\n")
            f.write("- [Raw Results](./raw_results.json)\n")
        
        # Generate summary visualizations
        if not df.empty and 'overall_score' in df.columns:
            fig = create_performance_chart(df, 'overall_score')
            plt.savefig(os.path.join(report_dir, "performance_chart.png"))
            plt.close()
            
            if cost_data:
                fig = create_cost_efficiency_chart(df, cost_data)
                plt.savefig(os.path.join(report_dir, "cost_efficiency.png"))
                plt.close()
        
        return summary_path
    
    def _generate_detailed_report(self, df: pd.DataFrame, 
                                 results: Dict[str, Dict[str, Any]],
                                 test_config: Dict[str, Any],
                                 report_dir: str,
                                 cost_data: Optional[Dict[str, float]] = None) -> str:
        """Generate detailed report with all metrics and visualizations"""
        from visualizations import create_radar_chart, create_heatmap
        
        detailed_path = os.path.join(report_dir, "detailed_metrics.md")
        
        with open(detailed_path, 'w') as f:
            f.write(f"# Detailed Model Comparison\n\n")
            
            # Test configuration
            f.write("## Test Configuration\n\n")
            f.write("```yaml\n")
            for key, value in test_config.items():
                f.write(f"{key}: {value}\n")
            f.write("```\n\n")
            
            # Per test case analysis
            f.write("## Test Case Analysis\n\n")
            
            test_cases = df['test_case'].unique()
            for test_case in test_cases:
                f.write(f"### {test_case}\n\n")
                
                test_df = df[df['test_case'] == test_case]
                f.write(test_df.to_markdown() + "\n\n")
                
                # Create radar chart for this test case if we have enough metrics
                numeric_cols = test_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 3:
                    metrics = [col for col in numeric_cols if col != 'overall_score'][:5]  # Limit to 5 metrics
                    if metrics:
                        fig = create_radar_chart(test_df, metrics)
                        radar_path = f"radar_{test_case.replace(' ', '_')}.png"
                        plt.savefig(os.path.join(report_dir, radar_path))
                        plt.close()
                        f.write(f"![Radar Chart](./{radar_path})\n\n")
            
            # Overall heatmap
            f.write("## Performance Heatmap\n\n")
            pivot_metrics = ['content_quality', 'formatting', 'creativity', 'consistency']
            available_metrics = [m for m in pivot_metrics if m in df.columns]
            
            if available_metrics:
                for metric in available_metrics:
                    fig = create_heatmap(df, 'model', 'test_case', metric)
                    heatmap_path = f"heatmap_{metric}.png"
                    plt.savefig(os.path.join(report_dir, heatmap_path))
                    plt.close()
                    f.write(f"### {metric.replace('_', ' ').title()} Heatmap\n\n")
                    f.write(f"![Heatmap](./{heatmap_path})\n\n")
        
        return detailed_path