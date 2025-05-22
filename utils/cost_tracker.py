
import json
import os
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from utils.logger import get_logger
from utils.file_utils import ensure_directory_exists, load_json, save_json

logger = get_logger(__name__)

class CostTracker:
    """
    Tracks API usage costs across different models and provides
    cost analysis and reporting.
    """
    
    def __init__(self, cost_file: str = "costs/api_costs.json"):
        """
        Initialize the cost tracker.
        
        Args:
            cost_file: Path to store cost data
        """
        self.cost_file = cost_file
        ensure_directory_exists(os.path.dirname(cost_file))
        
        # Load existing cost data if available
        self.cost_data = self._load_cost_data()
        
        # Current session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_costs: Dict[str, Dict[str, float]] = {}
    
    def _load_cost_data(self) -> Dict[str, Any]:
        """Load existing cost data or create new structure"""
        if os.path.exists(self.cost_file):
            try:
                return load_json(self.cost_file)
            except Exception as e:
                logger.error(f"Error loading cost data: {e}")
        
        # Default structure
        return {
            "sessions": {},
            "models": {},
            "total_cost": 0.0
        }
    
    def track_cost(self, model: str, tokens_in: int, tokens_out: int, 
                  test_case: Optional[str] = None) -> float:
        """
        Track cost for a single API call.
        
        Args:
            model: Model identifier
            tokens_in: Input tokens used
            tokens_out: Output tokens generated
            test_case: Optional test case identifier
            
        Returns:
            Cost for this API call
        """
        # Calculate cost based on model pricing
        cost = self._calculate_cost(model, tokens_in, tokens_out)
        
        # Update session tracking
        if model not in self.session_costs:
            self.session_costs[model] = {
                "cost": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "calls": 0,
                "test_cases": {}
            }
        
        self.session_costs[model]["cost"] += cost
        self.session_costs[model]["tokens_in"] += tokens_in
        self.session_costs[model]["tokens_out"] += tokens_out
        self.session_costs[model]["calls"] += 1
        
        # Track by test case if provided
        if test_case:
            if test_case not in self.session_costs[model]["test_cases"]:
                self.session_costs[model]["test_cases"][test_case] = {
                    "cost": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "calls": 0
                }
            
            self.session_costs[model]["test_cases"][test_case]["cost"] += cost
            self.session_costs[model]["test_cases"][test_case]["tokens_in"] += tokens_in
            self.session_costs[model]["test_cases"][test_case]["tokens_out"] += tokens_out
            self.session_costs[model]["test_cases"][test_case]["calls"] += 1
        
        logger.debug(f"Tracked cost for {model}: ${cost:.4f} ({tokens_in} in, {tokens_out} out)")
        return cost
    
    def _calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """
        Calculate cost based on model pricing.
        
        Args:
            model: Model identifier
            tokens_in: Input tokens
            tokens_out: Output tokens
            
        Returns:
            Calculated cost
        """
        # Pricing per 1000 tokens (input, output)
        # These are approximate and should be updated as pricing changes
        pricing = {
            # OpenAI models
            "gpt-4o": (5.0, 15.0),
            "gpt-4": (10.0, 30.0),
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-3.5-turbo": (0.5, 1.5),
            
            # Anthropic models
            "claude-3-opus": (15.0, 75.0),
            "claude-3-sonnet": (3.0, 15.0),
            "claude-3-haiku": (0.25, 1.25),
            
            # Google models
            "gemini-1.5-pro": (7.0, 21.0),
            "gemini-1.5-flash": (0.35, 1.05),
            
            # Mistral models
            "mistral-large": (8.0, 24.0),
            "mistral-medium": (2.7, 8.1),
            "mistral-small": (1.0, 3.0),
            
            # Meta models
            "llama-3-70b": (5.0, 15.0),
            "llama-3-8b": (0.7, 2.1),
        }
        
        # Default pricing if model not found
        default_pricing = (5.0, 15.0)
        
        # Get pricing for this model
        model_pricing = None
        for model_prefix, price in pricing.items():
            if model.startswith(model_prefix):
                model_pricing = price
                break
        
        if not model_pricing:
            logger.warning(f"No pricing found for model {model}, using default")
            model_pricing = default_pricing
        
        # Calculate cost
        input_cost = (tokens_in / 1000) * model_pricing[0]
        output_cost = (tokens_out / 1000) * model_pricing[1]
        
        return input_cost + output_cost
    
    def save_session(self) -> None:
        """Save the current session cost data"""
        # Update the cost data structure
        self.cost_data["sessions"][self.session_id] = {
            "timestamp": datetime.now().isoformat(),
            "models": self.session_costs,
            "total_cost": sum(model_data["cost"] for model_data in self.session_costs.values())
        }
        
        # Update model totals
        for model, model_data in self.session_costs.items():
            if model not in self.cost_data["models"]:
                self.cost_data["models"][model] = {
                    "cost": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "calls": 0
                }
            
            self.cost_data["models"][model]["cost"] += model_data["cost"]
            self.cost_data["models"][model]["tokens_in"] += model_data["tokens_in"]
            self.cost_data["models"][model]["tokens_out"] += model_data["tokens_out"]
            self.cost_data["models"][model]["calls"] += model_data["calls"]
        
        # Update total cost
        self.cost_data["total_cost"] = sum(model_data["cost"] for model_data in self.cost_data["models"].values())
        
        # Save to file
        save_json(self.cost_file, self.cost_data)
        logger.info(f"Saved cost data to {self.cost_file}")
    
    def get_session_costs(self) -> Dict[str, float]:
        """
        Get total costs for each model in the current session.
        
        Returns:
            Dictionary mapping model names to costs
        """
        return {model: data["cost"] for model, data in self.session_costs.items()}
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session costs.
        
        Returns:
            Dictionary with session cost summary
        """
        total_cost = sum(model_data["cost"] for model_data in self.session_costs.values())
        total_tokens_in = sum(model_data["tokens_in"] for model_data in self.session_costs.values())
        total_tokens_out = sum(model_data["tokens_out"] for model_data in self.session_costs.values())
        total_calls = sum(model_data["calls"] for model_data in self.session_costs.values())
        
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_cost": total_cost,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_calls": total_calls,
            "models": self.session_costs
        }
    
    def get_cost_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cost report.
        
        Returns:
            Dictionary with cost report data
        """
        # Calculate cost metrics
        total_cost = self.cost_data["total_cost"]
        model_costs = {model: data["cost"] for model, data in self.cost_data["models"].items()}
        
        # Sort models by cost
        sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate cost distribution
        cost_distribution = {}
        if total_cost > 0:
            cost_distribution = {model: (cost / total_cost) * 100 for model, cost in model_costs.items()}
        
        # Get session history
        sessions = list(self.cost_data["sessions"].values())
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "total_cost": total_cost,
            "model_costs": model_costs,
            "top_models": sorted_models[:5],
            "cost_distribution": cost_distribution,
            "recent_sessions": sessions[:10],
            "current_session": self.get_session_summary()
        }