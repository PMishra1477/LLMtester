
# evaluators/base_evaluator.py
from typing import Dict, List, Any
import re

class BaseEvaluator:
    """Base class for all evaluators"""

    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model response against test case requirements
        
        Args:
            response: Model response data
            test_case: Test case definition
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Basic evaluation common to all test types
        content = response.get("content", "")
        prompt = test_case.get("prompt", "")
        
        # Basic relevance check - how well does the response address the prompt
        relevance = self._evaluate_relevance(content, prompt, test_case)
        
        # Return basic metrics
        return {
            "relevance": relevance
        }
    
    def _evaluate_relevance(self, content: str, prompt: str, test_case: Dict[str, Any]) -> float:
        """Evaluate relevance of response to prompt"""
        # Extract key terms from prompt
        key_terms = self._extract_key_terms(prompt)
        
        if not key_terms:
            return 0.5  # Neutral score if no key terms
            
        # Count occurrences of key terms in response
        term_matches = 0
        for term in key_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                term_matches += 1
                
        # Calculate relevance score
        relevance_score = min(1.0, term_matches / len(key_terms)) if key_terms else 0.5
        
        # Consider expected elements if provided
        expected_elements = test_case.get("expected_elements", [])
        if expected_elements:
            element_matches = 0
            for element in expected_elements:
                # Convert element to regex pattern
                element_pattern = element.replace("_", "[ _-]")
                if re.search(r'\b' + element_pattern + r'\b', content, re.IGNORECASE):
                    element_matches += 1
            
            element_score = min(1.0, element_matches / len(expected_elements))
            # Combine with relevance score
            relevance_score = (relevance_score * 0.6) + (element_score * 0.4)
            
        return relevance_score
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common words and keep significant terms
        common_words = set([
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "about", "of", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
            "shall", "should", "may", "might", "must", "that", "which", "who", "whom",
            "whose", "what", "whatever", "when", "where", "how", "create", "make"
        ])
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if (
            word not in common_words and
            len(word) > 3 and  # Skip short words
            not word.isdigit()  # Skip numbers
        )]
        
        # Return unique terms
        return list(set(key_terms))
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple test case results
        
        Args:
            results: List of individual test case results
            
        Returns:
            Aggregated metrics
        """
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
        
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Get all metrics from first result
        first_result = successful_results[0]
        if "evaluation" not in first_result:
            return {"score": 0.0, "error": "No evaluation data"}
            
        metrics = list(first_result["evaluation"].keys())
        
        # Calculate average for each metric
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
                
        # Calculate overall score as simple average of all metrics
        aggregated["score"] = sum(aggregated.values()) / len(aggregated)
        
        # Add test metadata
        aggregated["total_tests"] = len(results)
        aggregated["successful_tests"] = len(successful_results)
        aggregated["failed_tests"] = len(results) - len(successful_results)
        
        return aggregated