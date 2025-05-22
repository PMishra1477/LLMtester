
# evaluators/creativity_evaluator.py
from typing import Dict, List, Any
import re
from evaluators.base_evaluator import BaseEvaluator

class CreativityEvaluator(BaseEvaluator):
    """Evaluator for creativity and innovation"""
    
    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for creativity metrics"""
        evaluation = super().evaluate(response, test_case)
        
        content = response.get("content", "")
        prompt = test_case.get("prompt", "")
        
        # Add creativity-specific metrics
        evaluation.update({
            "novelty": self._evaluate_novelty(content, prompt),
            "uniqueness": self._evaluate_uniqueness(content),
            "adaptability": self._evaluate_adaptability(content, test_case),
            "innovation": self._evaluate_innovation(content)
        })
        
        # Calculate overall score
        weights = {
            "relevance": 0.1,
            "novelty": 0.3,
            "uniqueness": 0.3,
            "adaptability": 0.2,
            "innovation": 0.1
        }
        
        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)
                          
        evaluation["score"] = weighted_sum / sum(weights.values())
        
        return evaluation
    
    def _evaluate_novelty(self, content: str, prompt: str) -> float:
        """Evaluate novelty (divergence from common responses)"""
        # Check for common phrases versus unique expressions
        common_phrases = [
            r"in conclusion",
            r"to summarize",
            r"as mentioned earlier",
            r"it is important to note",
            r"let me know if you have any questions"
        ]
        
        # Count common phrases
        common_count = 0
        for phrase in common_phrases:
            if re.search(phrase, content, re.IGNORECASE):
                common_count += 1
                
        # Check for prompt regurgitation
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Calculate overlap percentage
        if not prompt_words:
            overlap_percentage = 0
        else:
            overlap = prompt_words.intersection(content_words)
            overlap_percentage = len(overlap) / len(prompt_words)
        
        # Novel vocabulary (longer, less common words)
        long_words = re.findall(r'\b\w{8,}\b', content.lower())
        long_word_ratio = len(long_words) / len(content_words) if content_words else 0
        
        # Calculate novelty score
        common_phrase_penalty = min(0.5, common_count * 0.1)
        overlap_penalty = min(0.5, overlap_percentage * 0.5)
        vocabulary_bonus = min(0.3, long_word_ratio * 3)
        
        novelty_score = 0.7 - common_phrase_penalty - overlap_penalty + vocabulary_bonus
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, novelty_score))
    
    def _evaluate_uniqueness(self, content: str) -> float:
        """Evaluate uniqueness of approach and ideas"""
        # Uniqueness indicators
        unique_indicators = [
            r"(new|novel|innovative|unique|original|creative|different) (approach|idea|concept|perspective|viewpoint|angle|solution|method)",
            r"(unlike|different from|contrary to|as opposed to|in contrast to) (conventional|traditional|typical|standard|common|usual)",
            r"(breaking|break|broke) (away from|with|the mold|convention|tradition)",
            r"(think|thinking|thought) (outside|out of) the box",
            r"(first|pioneer|pioneering|revolutionary|groundbreaking)"
        ]
        
        # Count unique indicators
        unique_count = 0
        for pattern in unique_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            unique_count += len(matches)
            
        # Calculate base uniqueness score
        base_uniqueness = min(1.0, unique_count * 0.2)
        
        # Check for unusual word combinations
        words = re.findall(r'\b\w+\b', content.lower())
        word_pairs = [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]
        
        # Count unusual word pairs (simplified approximation)
        unusual_pairs = 0
        unusual_pair_patterns = [
            r"(sublime|ethereal|quantum|paradigm|zenith|enigmatic|serendipitous|ineffable|juxtapose) \w+",
            r"\w+ (symbiosis|dichotomy|paradox|alchemy|synthesis|metamorphosis|transcendence)"
        ]
        
        for pattern in unusual_pair_patterns:
            unusual_pairs += len(re.findall(pattern, content, re.IGNORECASE))
            
        pair_uniqueness = min(0.5, unusual_pairs * 0.1)
        
        return base_uniqueness + pair_uniqueness
    
    def _evaluate_adaptability(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate adaptability to constraints or requirements"""
        # Check for constraints in test case
        constraints = test_case.get("constraints", [])
        
        if not constraints:
            # Check for constraint-like terms in prompt
            prompt = test_case.get("prompt", "")
            constraint_terms = [
                "must", "should", "need to", "required", "limit", 
                "only", "exactly", "specific", "constraint"
            ]
            
            constraints = []
            for term in constraint_terms:
                if term in prompt.lower():
                    # Extract sentence containing constraint
                    sentences = re.split(r'[.!?]+', prompt)
                    for sentence in sentences:
                        if term in sentence.lower():
                            constraints.append(sentence.strip())
                            
        # If no constraints found or inferred, use default score
        if not constraints:
            return 0.7
            
        # Check compliance with each constraint
        compliance_count = 0
        for constraint in constraints:
            # Convert constraint to key terms
            key_terms = self._extract_key_terms(constraint)
            term_matches = 0
            
            for term in key_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                    term_matches += 1
                    
            # Consider constraint addressed if majority of terms present
            if term_matches > len(key_terms) / 2:
                compliance_count += 1
                
        adaptability_score = compliance_count / len(constraints) if constraints else 0.5
        
        return adaptability_score
    
    def _evaluate_innovation(self, content: str) -> float:
        """Evaluate innovation in approach or solution"""
        # Innovation indicators
        innovation_indicators = [
            # Solution patterns
            r"(novel|innovative|new|unique) (solution|approach|method|technique)",
            r"(solves|addresses|tackles|resolves|mitigates) (problem|issue|challenge|difficulty)",
            
            # Improvement patterns
            r"(improve|enhance|optimize|upgrade|elevate) (existing|current|conventional|traditional)",
            r"(better than|superior to|more effective than) (existing|current|conventional|traditional)",
            
            # Perspective patterns
            r"(alternative|different|fresh|new) (perspective|viewpoint|angle|lens|approach)",
            r"(reframe|rethink|reimagine|reconsider) (problem|issue|challenge|approach)"
        ]
        
        # Count innovation indicators
        innovation_count = 0
        for pattern in innovation_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            innovation_count += len(matches)
            
        # Calculate innovation score
        base_innovation = min(1.0, innovation_count * 0.2)
        
        # Check for solution specificity
        solution_patterns = [
            r"(specifically|in particular|concretely)",
            r"(works by|functions by|operates by|achieves this by)",
            r"(step \d|first step|second step|next step|final step)"
        ]
        
        specificity_count = 0
        for pattern in solution_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            specificity_count += len(matches)
            
        specificity_score = min(0.5, specificity_count * 0.1)
        
        return base_innovation + specificity_score
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for creativity evaluation"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
            
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Extract metrics from each result
        metrics = [
            "relevance",
            "novelty",
            "uniqueness",
            "adaptability",
            "innovation",
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