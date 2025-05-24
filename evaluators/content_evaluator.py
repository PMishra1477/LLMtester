
# evaluators/content_evaluator.py
from typing import Dict, List, Any
import re
from evaluators.base_evaluator import BaseEvaluator

class ContentEvaluator(BaseEvaluator):
    """Evaluator for content quality"""

    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for content quality metrics"""
        evaluation = super().evaluate(response, test_case)
        
        content = response.get("content", "")
        
        # Add content-specific metrics
        evaluation.update({
            "accuracy": self._evaluate_accuracy(content, test_case),
            "comprehensiveness": self._evaluate_comprehensiveness(content, test_case),
            "depth": self._evaluate_depth(content),
            "clarity": self._evaluate_clarity(content)
        })
        
        # Calculate overall score
        weights = {
            "relevance": 0.2,
            "accuracy": 0.25,
            "comprehensiveness": 0.2,
            "depth": 0.15,
            "clarity": 0.2
        }
        
        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)
                          
        evaluation["score"] = weighted_sum / sum(weights.values())
        
        return evaluation
    
    def _evaluate_accuracy(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate factual accuracy of content"""
        # Check against reference data if available
        reference_data = test_case.get("reference_data", {})
        if not reference_data:
            return 0.7  # Default score when no reference data
            
        correct_facts = 0
        total_facts = 0
        
        # Check for each fact in reference data
        for category, facts in reference_data.items():
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, dict):
                        # Handle dictionary format (e.g., {"name": "Mercury", "distance": 0.39})
                        for key, value in fact.items():
                            total_facts += 1
                            fact_pattern = f"{fact.get('name', '')}.+{key}.+{value}"
                            if re.search(fact_pattern, content, re.IGNORECASE | re.DOTALL):
                                correct_facts += 1
                    else:
                        # Handle simple facts
                        total_facts += 1
                        if re.search(r'\b' + re.escape(str(fact)) + r'\b', content, re.IGNORECASE):
                            correct_facts += 1
            elif isinstance(facts, dict):
                # Handle dictionary directly
                for key, value in facts.items():
                    total_facts += 1
                    fact_pattern = f"{key}.+{value}"
                    if re.search(fact_pattern, content, re.IGNORECASE | re.DOTALL):
                        correct_facts += 1
        
        if total_facts == 0:
            return 0.7  # Default when no facts to check
            
        return correct_facts / total_facts
    
    def _evaluate_comprehensiveness(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate how comprehensive the content is"""
        # Check for expected elements
        expected_elements = test_case.get("expected_elements", [])
        if not expected_elements:
            # Fallback to estimating comprehensiveness by content length and sections
            words = len(content.split())
            sections = len(re.findall(r"#{1,3}\s+", content)) + 1  # Add 1 for main content
            
            # Score based on length and structure
            length_score = min(1.0, words / 1000)  # Assume 1000 words is comprehensive
            section_score = min(1.0, sections / 5)  # Assume 5 sections is comprehensive
            
            return (length_score * 0.7) + (section_score * 0.3)
        
        # Count covered elements
        covered = 0
        for element in expected_elements:
            element_pattern = element.replace("_", "[ _-]")
            if re.search(r'\b' + element_pattern + r'\b', content, re.IGNORECASE):
                covered += 1
                
        return covered / len(expected_elements) if expected_elements else 0.5
    
    def _evaluate_depth(self, content: str) -> float:
        """Evaluate depth of analysis or explanation"""
        # Depth indicators
        depth_indicators = [
            # Explanation patterns
            r"because",
            r"due to",
            r"as a result of",
            r"the reason is",
            r"this is important",
            r"contributes to",
            
            # Analysis patterns
            r"analysis",
            r"in comparison",
            r"on the other hand",
            r"alternatively",
            r"in contrast",
            
            # Detail patterns
            r"specifically",
            r"for example",
            r"for instance",
            r"such as",
            r"including",
            
            # Technical depth
            r"technically",
            r"in technical terms",
            r"the process",
            r"the mechanism"
        ]
        
        # Count depth indicators
        depth_count = 0
        for pattern in depth_indicators:
            matches = re.findall(r"\b" + pattern + r"\b", content, re.IGNORECASE)
            depth_count += len(matches)
            
        # Normalize by content length
        words = len(content.split())
        if words == 0:
            return 0.0
            
        # Calculate depth density
        depth_density = depth_count / (words / 100)  # Per 100 words
        
        # Score based on density
        return min(1.0, depth_density / 2)  # Assume 2 depth indicators per 100 words is good
    
    def _evaluate_clarity(self, content: str) -> float:
        """Evaluate clarity and readability of content"""
        # Clarity indicators
        clarity_indicators = {
            # Positive indicators
            "positive": [
                r"clear",
                r"concise",
                r"straightforward",
                r"in summary",
                r"to summarize",
                r"in conclusion"
            ],
            
            # Negative indicators
            "negative": [
                r"however,[^.]{50,}",  # Long sentence after "however"
                r"[^.]{60,}",  # Very long sentences
                r"[a-z]{15,}",  # Very long words
                r"(that that|and and)",  # Repeated words
                r"[\(\[\{][^\)\]\}]{100,}[\)\]\}]"  # Very long parenthetical
            ]
        }
        
        # Count indicators
        positive_count = 0
        for pattern in clarity_indicators["positive"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            positive_count += len(matches)
            
        negative_count = 0
        for pattern in clarity_indicators["negative"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            negative_count += len(matches)
            
        # Calculate base readability metrics
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        
        if words == 0 or sentences == 0:
            return 0.5
            
        avg_words_per_sentence = words / sentences
        
        # Calculate clarity score
        sentence_length_penalty = max(0, min(1, (30 - avg_words_per_sentence) / 20))
        positive_score = min(1.0, positive_count / 5)  # Assume 5 positive indicators is good
        negative_penalty = max(0, min(0.5, negative_count / 5))  # Cap penalty at 0.5
        
        clarity_score = (sentence_length_penalty * 0.4) + (positive_score * 0.3) + (1 - negative_penalty) * 0.3
        
        return clarity_score
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for content evaluation"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
            
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Extract metrics from each result
        metrics = [
            "relevance",
            "accuracy",
            "comprehensiveness",
            "depth",
            "clarity",
            "score"
        ]
        
        aggregated = {}
        for metric in metrics:
            values = [
                r["evaluation"][metric]
                for r in successful_results
                if "evaluation" in r and metric in r["evaluation"]
            ]
            
            if values:
                aggregated[metric] = sum(values) / len(values)
            else:
                aggregated[metric] = 0.0
                
        # Add test metadata
        aggregated["total_tests"] = len(results)
        aggregated["successful_tests"] = len(successful_results)
        aggregated["failed_tests"] = len(results) - len(successful_results)
        
        return aggregated