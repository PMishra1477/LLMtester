
# evaluators/efficiency_evaluator.py
from typing import Dict, List, Any
import re
from evaluators.base_evaluator import BaseEvaluator

class EfficiencyEvaluator(BaseEvaluator):
    """Evaluator for efficiency metrics like token usage and instruction following"""
    
    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for efficiency metrics"""
        evaluation = super().evaluate(response, test_case)
        
        content = response.get("content", "")
        prompt = test_case.get("prompt", "")
        usage = response.get("usage", {})
        
        # Add efficiency-specific metrics
        evaluation.update({
            "token_usage": self._evaluate_token_usage(content, usage),
            "instruction_following": self._evaluate_instruction_following(content, prompt, test_case),
            "completeness": self._evaluate_completeness(content, test_case)
        })
        
        # Calculate overall score
        weights = {
            "relevance": 0.2,
            "token_usage": 0.3,
            "instruction_following": 0.3,
            "completeness": 0.2
        }
        
        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)
                          
        evaluation["score"] = weighted_sum / sum(weights.values())
        
        return evaluation
    
    def _evaluate_token_usage(self, content: str, usage: Dict[str, Any]) -> float:
        """Evaluate efficiency of token usage"""
        # Get token counts
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        if output_tokens == 0:
            # Estimate tokens if not provided
            output_tokens = len(content.split()) * 1.3  # Rough token estimate
            
        # Calculate information density (meaningful words to total tokens)
        words = content.split()
        
        # Count meaningful words (exclude common words)
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                       "in", "on", "at", "to", "for", "with", "by", "about", "like",
                       "from", "as", "of", "that", "this", "these", "those", "it", "its"}
        
        meaningful_words = [w for w in words if w.lower() not in common_words]
        meaningful_ratio = len(meaningful_words) / len(words) if words else 0
        
        # Evaluate output token efficiency
        content_length = len(content)
        if content_length == 0:
            return 0.0
            
        # Calculate token efficiency (higher is better)
        content_per_token = content_length / output_tokens if output_tokens > 0 else 0
        
        # Normalize to a score between 0 and 1
        # Assume ideal is around 5-6 characters per token
        efficiency_score = min(1.0, content_per_token / 5.5) 
        
        # Adjust based on meaningful content ratio
        adjusted_score = efficiency_score * (0.5 + (meaningful_ratio * 0.5))
        
        return adjusted_score
    
    def _evaluate_instruction_following(self, content: str, prompt: str, test_case: Dict[str, Any]) -> float:
        """Evaluate how well instructions were followed"""
        # Extract instructions from prompt
        instruction_patterns = [
            r"(create|generate|make|produce|provide|give me|develop) (a|an) ([^.]+)",
            r"(include|add|incorporate|ensure|make sure) ([^.]+)",
            r"(do not|don't|avoid|exclude) ([^.]+)",
            r"(use|utilize|employ|apply) ([^.]+)",
            r"(follow|adhere to|comply with|conform to) ([^.]+)"
        ]
        
        instructions = []
        for pattern in instruction_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            instructions.extend(matches)
            
        if not instructions:
            return 0.7  # Default when no clear instructions found
            
        # Check compliance with instructions
        compliance_count = 0
        
        for instruction in instructions:
            # Convert instruction tuple to string
            if isinstance(instruction, tuple):
                instruction_text = " ".join(instruction)
            else:
                instruction_text = instruction
                
            # Extract key terms
            key_terms = self._extract_key_terms(instruction_text)
            term_matches = 0
            
            for term in key_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                    term_matches += 1
                    
            # Consider instruction followed if majority of terms present
            if term_matches > len(key_terms) / 2:
                compliance_count += 1
                
        # Calculate instruction following score
        if not instructions:
            return 0.7  # Default score
        return compliance_count / len(instructions)
    
    def _evaluate_completeness(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate completeness of response"""
        # Check for expected elements
        expected_elements = test_case.get("expected_elements", [])
        
        if not expected_elements:
            # Estimate completeness by length and structure
            expected_length = test_case.get("parameters", {}).get("max_tokens", 1000) / 2
            actual_words = len(content.split())
            
            # Calculate length-based completeness (cap at 100%)
            length_score = min(1.0, actual_words / expected_length)
            
            # Check for potential truncation
            truncation_indicators = [
                r"\.{3,}$",  # Ending with ellipsis
                r"[a-z]$",   # Ending with lowercase (no period)
                r"^[A-Z]"    # Starting with uppercase (continuation)
            ]
            
            is_truncated = any(re.search(pattern, content) for pattern in truncation_indicators)
            
            return length_score * (0.7 if is_truncated else 1.0)
            
        # Count covered elements
        covered = 0
        for element in expected_elements:
            element_pattern = element.replace("_", "[ _-]")
            if re.search(r'\b' + element_pattern + r'\b', content, re.IGNORECASE):
                covered += 1
                
        return covered / len(expected_elements) if expected_elements else 0.7
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for efficiency evaluation"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
            
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Extract metrics from each result
        metrics = [
            "relevance",
            "token_usage",
            "instruction_following",
            "completeness",
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