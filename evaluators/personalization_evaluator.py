
# evaluators/personalization_evaluator.py
from typing import Dict, List, Any
import re
from evaluators.base_evaluator import BaseEvaluator

class PersonalizationEvaluator(BaseEvaluator):
    """Evaluator for personalization and audience targeting"""
    
    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for personalization metrics"""
        evaluation = super().evaluate(response, test_case)
        
        content = response.get("content", "")
        
        # Add personalization-specific metrics
        evaluation.update({
            "audience_targeting": self._evaluate_audience_targeting(content, test_case),
            "industry_adaptation": self._evaluate_industry_adaptation(content, test_case),
            "customization": self._evaluate_customization(content, test_case)
        })
        
        # Calculate overall score
        weights = {
            "relevance": 0.1,
            "audience_targeting": 0.3,
            "industry_adaptation": 0.3,
            "customization": 0.3
        }
        
        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)
                          
        evaluation["score"] = weighted_sum / sum(weights.values())
        
        return evaluation
    
    def _evaluate_audience_targeting(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate how well content targets specified audience"""
        # Extract audience from test case
        audience = test_case.get("audience", "")
        
        if not audience:
            # Try to extract from prompt
            prompt = test_case.get("prompt", "")
            audience_patterns = [
                r"for (an audience of|a group of|a team of) ([^.]+)",
                r"(targeting|aimed at|designed for|intended for) ([^.]+)",
                r"(audience|viewers|readers|users) (are|is|includes|consists of) ([^.]+)"
            ]
            
            for pattern in audience_patterns:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    # Extract audience from first match
                    if isinstance(matches[0], tuple):
                        audience = matches[0][-1]  # Last group contains the audience
                    else:
                        audience = matches[0]
                    break
                    
        if not audience:
            return 0.7  # Default when no audience specified
            
        # Check for audience-specific language
        audience_terms = self._extract_key_terms(audience)
        direct_mentions = 0
        
        for term in audience_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                direct_mentions += 1
                
        direct_mention_score = min(0.5, (direct_mentions / len(audience_terms)) * 0.5) if audience_terms else 0.3
        
        # Check for audience-appropriate content adaptations
        adaptations_score = 0.0
        
        # Common audience types and their expected adaptations
        audience_adaptations = {
            "technical": ["technical terms", "specifications", "detailed explanation", "methodology"],
            "executive": ["summary", "key points", "strategic", "overview", "bottom line"],
            "beginner": ["introduction", "basics", "fundamentals", "simple terms", "step by step"],
            "children": ["simple", "fun", "engaging", "colorful", "easy to understand"],
            "customer": ["benefits", "value", "features", "solution", "results"],
            "investor": ["roi", "growth", "market", "opportunity", "returns"]
        }
        
        # Check if audience matches any of the common types
        matched_audiences = []
        for audience_type in audience_adaptations:
            if audience_type.lower() in audience.lower():
                matched_audiences.append(audience_type)
                
        # If no direct matches, try keyword matching
        if not matched_audiences:
            technical_indicators = ["developer", "engineer", "scientist", "programmer", "technical"]
            executive_indicators = ["executive", "ceo", "manager", "director", "leadership"]
            beginner_indicators = ["beginner", "novice", "new", "learning", "student"]
            children_indicators = ["child", "kid", "young", "elementary", "school"]
            customer_indicators = ["customer", "client", "consumer", "user", "buyer"]
            investor_indicators = ["investor", "stakeholder", "shareholder", "venture", "capital"]
            
            audience_lower = audience.lower()
            
            if any(ind in audience_lower for ind in technical_indicators):
                matched_audiences.append("technical")
            if any(ind in audience_lower for ind in executive_indicators):
                matched_audiences.append("executive")
            if any(ind in audience_lower for ind in beginner_indicators):
                matched_audiences.append("beginner")
            if any(ind in audience_lower for ind in children_indicators):
                matched_audiences.append("children")
            if any(ind in audience_lower for ind in customer_indicators):
                matched_audiences.append("customer")
            if any(ind in audience_lower for ind in investor_indicators):
                matched_audiences.append("investor")
                
        # Check for presence of audience-appropriate adaptations
        if matched_audiences:
            adaptation_matches = 0
            adaptation_total = 0
            
            for matched in matched_audiences:
                adaptation_terms = audience_adaptations[matched]
                
                for term in adaptation_terms:
                    adaptation_total += 1
                    if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                        adaptation_matches += 1
                        
            adaptations_score = adaptation_matches / adaptation_total if adaptation_total > 0 else 0.2
        else:
            # Generic adaptation indicators
            adaptations_score = 0.2  # Default for unknown audience
            
        # Check for second-person language (you/your) indicating audience awareness
        second_person_count = len(re.findall(r'\b(you|your|yours)\b', content, re.IGNORECASE))
        second_person_density = second_person_count / len(content.split()) if content.split() else 0
        second_person_score = min(0.3, second_person_density * 10)
        
        return direct_mention_score + adaptations_score + second_person_score
    
    def _evaluate_industry_adaptation(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate how well content adapts to specific industry"""
        # Extract industry from test case
        industry = test_case.get("industry", "")
        
        if not industry:
            # Try to extract from prompt
            prompt = test_case.get("prompt", "")
            industry_patterns = [
                r"in the ([\w\s-]+) (industry|sector|field|domain)",
                r"for (the|a|an) ([\w\s-]+) (company|business|organization|firm)",
                r"(related to|pertaining to|concerning|about) (the|a|an) ([\w\s-]+) (industry|sector)"
            ]
            
            for pattern in industry_patterns:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    # Extract industry from match
                    if isinstance(matches[0], tuple):
                        # Get the industry name from the tuple
                        if len(matches[0]) >= 2:
                            industry = matches[0][0] if "industry" in pattern else matches[0][1]
                    else:
                        industry = matches[0]
                    break
                    
        if not industry:
            return 0.7  # Default when no industry specified
            
        # Check for industry-specific terminology
        industry_terms = self._extract_key_terms(industry)
        direct_mentions = 0
        
        for term in industry_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                direct_mentions += 1
                
        direct_mention_score = min(0.4, (direct_mentions / len(industry_terms)) * 0.4) if industry_terms else 0.2
        
        # Industry-specific jargon lookup
        industry_jargon = {
            "technology": ["infrastructure", "scalability", "interface", "deployment", "integration", "API"],
            "finance": ["portfolio", "assets", "diversification", "equity", "liquidity", "yield"],
            "healthcare": ["patient", "clinical", "diagnosis", "treatment", "care", "medical"],
            "education": ["curriculum", "learning", "student", "teaching", "assessment", "education"],
            "retail": ["consumer", "inventory", "merchandising", "shopping", "pricing", "store"],
            "manufacturing": ["production", "assembly", "quality", "supply chain", "raw materials", "facility"],
            "real estate": ["property", "listing", "market value", "buyer", "seller", "investment"],
            "marketing": ["branding", "campaign", "audience", "conversion", "engagement", "metrics"]
        }
        
        # Check if the industry matches any in our jargon dictionary
        jargon_score = 0.0
        matched_industry = None
        
        for ind, terms in industry_jargon.items():
            if ind.lower() in industry.lower():
                matched_industry = ind
                break
                
        if matched_industry:
            jargon_terms = industry_jargon[matched_industry]
            jargon_matches = 0
            
            for term in jargon_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                    jargon_matches += 1
                    
            jargon_score = min(0.4, (jargon_matches / len(jargon_terms)) * 0.4)
        else:
            # Generic industry adaptations
            industry_indicators = ["industry", "sector", "field", "market", "business", "professional"]
            
            indicator_matches = 0
            for indicator in industry_indicators:
                if re.search(r'\b' + re.escape(indicator) + r'\b', content, re.IGNORECASE):
                    indicator_matches += 1
                    
            jargon_score = min(0.2, (indicator_matches / len(industry_indicators)) * 0.2)
            
        # Check for industry-appropriate examples
        examples_pattern = r"(for (example|instance)|such as|e\.g\.|including)"
        has_examples = bool(re.search(examples_pattern, content, re.IGNORECASE))
        examples_score = 0.2 if has_examples else 0.0
        
        return direct_mention_score + jargon_score + examples_score
    
    def _evaluate_customization(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate how well content is customized to specific requirements"""
        # Extract customization requirements
        requirements = test_case.get("requirements", [])
        
        if not requirements:
            # Look for requirement-like elements in prompt
            prompt = test_case.get("prompt", "")
            requirement_patterns = [
                r"(must|should|need to|have to) (include|contain|incorporate|address) ([^.]+)",
                r"(please|kindly) (ensure|make sure|include|add) ([^.]+)",
                r"(requirements?|specifications?|guidelines?|instructions?):\s*([^.]+)"
            ]
            
            for pattern in requirement_patterns:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    # Extract requirements from matches
                    for match in matches:
                        if isinstance(match, tuple):
                            req = match[-1]  # Last group contains the requirement
                            requirements.append(req)
                            
        if not requirements:
            return 0.7  # Default when no requirements specified
            
        # Check each requirement
        requirement_scores = []
        
        for req in requirements:
            # Extract key terms
            req_terms = self._extract_key_terms(req)
            matches = 0
            
            for term in req_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                    matches += 1
                    
            # Calculate score for this requirement
            if req_terms:
                requirement_scores.append(matches / len(req_terms))
            else:
                requirement_scores.append(0.5)  # Default for empty requirements
                
        # Average requirement scores
        if not requirement_scores:
            return 0.7
            
        return sum(requirement_scores) / len(requirement_scores)
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for personalization evaluation"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
            
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Extract metrics from each result
        metrics = [
            "relevance",
            "audience_targeting",
            "industry_adaptation",
            "customization",
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