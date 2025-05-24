
# evaluators/consistency_evaluator.py
from typing import Dict, List, Any
import re
from evaluators.base_evaluator import BaseEvaluator

class ConsistencyEvaluator(BaseEvaluator):
    """Evaluator for consistency in style, tone, and branding"""
    
    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for consistency metrics"""
        evaluation = super().evaluate(response, test_case)
        
        content = response.get("content", "")
        
        # Add consistency-specific metrics
        evaluation.update({
            "style_consistency": self._evaluate_style_consistency(content),
            "tone_consistency": self._evaluate_tone_consistency(content),
            "branding_consistency": self._evaluate_branding_consistency(content, test_case)
        })
        
        # Calculate overall score
        weights = {
            "relevance": 0.1,
            "style_consistency": 0.3,
            "tone_consistency": 0.3,
            "branding_consistency": 0.3
        }
        
        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)
                          
        evaluation["score"] = weighted_sum / sum(weights.values())
        
        return evaluation
    
    def _evaluate_style_consistency(self, content: str) -> float:
        """Evaluate consistency of writing style"""
        # Break content into paragraphs
        paragraphs = re.split(r"\n\s*\n", content)
        if len(paragraphs) <= 1:
            return 0.7  # Default for short content
            
        # Analyze style indicators across paragraphs
        style_indicators = {
            "sentence_length": [],
            "punctuation_density": [],
            "question_frequency": [],
            "formal_language": []
        }
        
        for para in paragraphs:
            # Skip very short paragraphs
            if len(para.split()) < 10:
                continue
                
            # Count sentences
            sentences = re.split(r'[.!?]+', para)
            sentences = [s for s in sentences if s.strip()]
            
            if not sentences:
                continue
                
            # Average sentence length
            words_per_sentence = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(words_per_sentence) / len(words_per_sentence)
            style_indicators["sentence_length"].append(avg_sentence_length)
            
            # Punctuation density
            punctuation_count = len(re.findall(r'[,.;:!?]', para))
            word_count = len(para.split())
            punctuation_density = punctuation_count / word_count if word_count > 0 else 0
            style_indicators["punctuation_density"].append(punctuation_density)
            
            # Question frequency
            question_count = len(re.findall(r'\?', para))
            question_ratio = question_count / len(sentences) if sentences else 0
            style_indicators["question_frequency"].append(question_ratio)
            
            # Formal language indicators
            formal_indicators = [
                "therefore", "thus", "consequently", "furthermore", "moreover",
                "however", "nevertheless", "accordingly", "subsequently", "hereby"
            ]
            
            formal_count = sum(1 for word in formal_indicators if word in para.lower())
            formal_ratio = formal_count / word_count if word_count > 0 else 0
            style_indicators["formal_language"].append(formal_ratio)
            
        # Calculate variance for each indicator
        consistency_scores = []
        for indicator, values in style_indicators.items():
            if not values:
                continue
                
            # Calculate variance (normalized by mean to get coefficient of variation)
            mean = sum(values) / len(values)
            if mean == 0:
                continue
                
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            coefficient_of_variation = (variance ** 0.5) / mean
            
            # Convert to consistency score (lower variation = higher consistency)
            indicator_consistency = max(0, min(1, 1 - (coefficient_of_variation * 2)))
            consistency_scores.append(indicator_consistency)
            
        if not consistency_scores:
            return 0.5  # Default when not enough data
            
        return sum(consistency_scores) / len(consistency_scores)
    
    def _evaluate_tone_consistency(self, content: str) -> float:
        """Evaluate consistency of tone throughout content"""
        # Tone categories and their indicators
        tone_categories = {
            "formal": [
                "furthermore", "moreover", "therefore", "thus", "consequently",
                "accordingly", "subsequently", "indeed", "precisely", "necessarily"
            ],
            "casual": [
                "anyway", "basically", "actually", "honestly", "literally",
                "like", "so", "pretty", "kind of", "sort of", "you know"
            ],
            "technical": [
                "specifically", "particularly", "notably", "significantly",
                "technically", "fundamentally", "essentially", "predominantly"
            ],
            "enthusiastic": [
                "amazing", "awesome", "fantastic", "incredible", "excellent",
                "wonderful", "great", "exciting", "thrilling", "extraordinary"
            ],
            "cautious": [
                "perhaps", "possibly", "potentially", "may", "might",
                "could", "seemingly", "apparently", "ostensibly", "presumably"
            ]
        }
        
        # Break content into sections
        sections = re.split(r"\n\s*\n|#{1,6}\s+", content)
        sections = [s for s in sections if len(s.split()) >= 15]  # Skip very short sections
        
        if len(sections) <= 1:
            return 0.7  # Default for short content
            
        # Analyze tone in each section
        section_tones = []
        
        for section in sections:
            section_lower = section.lower()
            tone_scores = {}
            
            # Count tone indicators
            for tone, indicators in tone_categories.items():
                count = sum(1 for indicator in indicators if re.search(r'\b' + re.escape(indicator) + r'\b', section_lower))
                # Normalize by section length
                word_count = len(section_lower.split())
                tone_scores[tone] = count / (word_count / 100)  # Per 100 words
                
            # Determine dominant tone
            if tone_scores:
                dominant_tone = max(tone_scores, key=tone_scores.get)
                section_tones.append(dominant_tone)
                
        if not section_tones:
            return 0.5  # Default when not enough data
            
        # Check consistency of dominant tone
        tone_counts = {}
        for tone in section_tones:
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
            
        # Calculate tone consistency
        most_common_tone_count = max(tone_counts.values())
        tone_consistency = most_common_tone_count / len(section_tones)
        
        return tone_consistency
    
    def _evaluate_branding_consistency(self, content: str, test_case: Dict[str, Any]) -> float:
        """Evaluate consistency of branding elements"""
        # Check for brand elements in test case
        brand_elements = test_case.get("brand_elements", {})
        
        if not brand_elements:
            # Look for brand mentions in prompt
            prompt = test_case.get("prompt", "")
            brand_mentions = re.findall(r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b', prompt)
            potential_brands = [b for b in brand_mentions if len(b) >= 4]
            
            if potential_brands:
                # Use the most frequently mentioned potential brand
                brand_counts = {}
                for brand in potential_brands:
                    brand_counts[brand] = brand_counts.get(brand, 0) + 1
                
                top_brand = max(brand_counts, key=brand_counts.get)
                brand_elements = {"name": top_brand}
                
        if not brand_elements:
            return 0.7  # Default when no brand elements identified
            
        # Check for brand name consistency
        brand_name = brand_elements.get("name", "")
        if brand_name:
            brand_mentions = len(re.findall(r'\b' + re.escape(brand_name) + r'\b', content, re.IGNORECASE))
            
            # Calculate expected mentions based on length
            word_count = len(content.split())
            expected_mentions = max(1, word_count / 300)  # Expect mention every ~300 words
            
            name_consistency = min(1.0, brand_mentions / expected_mentions)
        else:
            name_consistency = 0.5
            
        # Check for other brand elements
        slogan = brand_elements.get("slogan", "")
        slogan_included = 1.0 if slogan and slogan.lower() in content.lower() else 0.0
        
        colors = brand_elements.get("colors", [])
        color_mentions = sum(1 for color in colors if re.search(r'\b' + re.escape(color) + r'\b', content, re.IGNORECASE))
        color_consistency = min(1.0, color_mentions / len(colors)) if colors else 0.5
        
        # Calculate overall branding consistency
        brand_elements_count = sum(1 for e in [brand_name, slogan, colors] if e)
        
        if brand_elements_count == 0:
            return 0.7  # Default when no brand elements
            
        weights = []
        scores = []
        
        if brand_name:
            weights.append(0.6)
            scores.append(name_consistency)
            
        if slogan:
            weights.append(0.2)
            scores.append(slogan_included)
            
        if colors:
            weights.append(0.2)
            scores.append(color_consistency)
            
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        return weighted_sum / sum(weights)
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for consistency evaluation"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
            
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Extract metrics from each result
        metrics = [
            "relevance",
            "style_consistency",
            "tone_consistency",
            "branding_consistency",
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