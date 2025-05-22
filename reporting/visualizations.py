
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

def create_performance_chart(df: pd.DataFrame, metric: str = 'overall_score') -> plt.Figure:
    """
    Create a bar chart comparing model performance on a specific metric.
    
    Args:
        df: DataFrame with test results
        metric: Metric to visualize
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate average scores by model
    avg_scores = df.groupby('model')[metric].mean().sort_values(ascending=False)
    
    # Create bar chart
    ax = avg_scores.plot(kind='bar', color='skyblue')
    plt.title(f'Model Performance Comparison: {metric.replace("_", " ").title()}')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(avg_scores):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def create_cost_efficiency_chart(df: pd.DataFrame, 
                                cost_data: Dict[str, float],
                                metric: str = 'overall_score') -> plt.Figure:
    """
    Create a chart showing cost efficiency (performance per dollar).
    
    Args:
        df: DataFrame with test results
        cost_data: Dictionary mapping model names to total costs
        metric: Performance metric to use
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate average scores by model
    avg_scores = df.groupby('model')[metric].mean()
    
    # Calculate efficiency (score per dollar)
    efficiency = {}
    for model in avg_scores.index:
        if model in cost_data and cost_data[model] > 0:
            efficiency[model] = avg_scores[model] / cost_data[model]
        else:
            efficiency[model] = 0
    
    efficiency_series = pd.Series(efficiency).sort_values(ascending=False)
    
    # Create bar chart
    ax = efficiency_series.plot(kind='bar', color='lightgreen')
    plt.title('Cost Efficiency: Performance per Dollar')
    plt.ylabel(f'{metric.replace("_", " ").title()} per Dollar')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(efficiency_series):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def create_radar_chart(df: pd.DataFrame, metrics: List[str]) -> plt.Figure:
    """
    Create a radar chart comparing models across multiple metrics.
    
    Args:
        df: DataFrame with test results
        metrics: List of metrics to include in the radar chart
        
    Returns:
        Matplotlib figure
    """
    # Ensure we have valid metrics
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < 3:
        logger.warning("Not enough metrics for radar chart")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Not enough metrics for radar chart", 
                ha='center', va='center')
        return fig
    
    # Calculate average scores by model for each metric
    avg_scores = df.groupby('model')[available_metrics].mean()
    
    # Number of metrics
    N = len(available_metrics)
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Add metric labels
    metric_labels = [m.replace('_', ' ').title() for m in available_metrics]
    plt.xticks(angles[:-1], metric_labels, size=12)
    
    # Plot each model
    for model in avg_scores.index:
        values = avg_scores.loc[model].values.flatten().tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison Across Metrics', size=15)
    
    return fig

def create_heatmap(df: pd.DataFrame, row: str, col: str, value: str) -> plt.Figure:
    """
    Create a heatmap visualization.
    
    Args:
        df: DataFrame with test results
        row: Column to use for rows
        col: Column to use for columns
        value: Column to use for cell values
        
    Returns:
        Matplotlib figure
    """
    # Create pivot table
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc='mean')
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
    plt.title(f'{value.replace("_", " ").title()} by {row.title()} and {col.replace("_", " ").title()}')
    plt.tight_layout()
    
    return plt.gcf()