
# evaluators/evaluator_factory.py
from typing import Dict, Any

class EvaluatorFactory:
    """Factory for creating evaluators based on test category"""

    def __init__(self):
        self.evaluators = {}

    def get_evaluator(self, category: str) -> Any:
        """Get or create an evaluator for the specified category"""
        if category not in self.evaluators:
            evaluator_class = self._get_evaluator_class(category)
            self.evaluators[category] = evaluator_class()

        return self.evaluators[category]

    def _get_evaluator_class(self, category: str) -> type:
        """Get the appropriate evaluator class for the category"""
        import importlib

        category_map = {
            "ppt": "PPTEvaluator",
            "content": "ContentEvaluator",
            "creativity": "CreativityEvaluator",
            "efficiency": "EfficiencyEvaluator",
            "formatting": "FormattingEvaluator",
            "consistency": "ConsistencyEvaluator",
            "personalization": "PersonalizationEvaluator"
        }

        if category in category_map:
            class_name = category_map[category]
        else:
            # Default to base evaluator
            class_name = "BaseEvaluator"

        try:
            module = importlib.import_module(f"evaluators.{category}_evaluator")
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            # Fallback to base evaluator
            from evaluators.base_evaluator import BaseEvaluator
            return BaseEvaluator