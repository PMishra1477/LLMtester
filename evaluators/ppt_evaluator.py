# evaluators/ppt_evaluator.py
from typing import Dict, List, Any
import re
import json
from evaluators.base_evaluator import BaseEvaluator

class PPTEvaluator(BaseEvaluator):
    """Evaluator for PowerPoint generation capabilities"""

    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for PPT-specific metrics"""
        evaluation = super().evaluate(response, test_case)

        # Get the generated content
        content = response.get("content", "")

        # Add PPT-specific metrics
        evaluation.update({
            "slide_structure": self._evaluate_slide_structure(content),
            "visual_elements": self._evaluate_visual_elements(content),
            "slide_transitions": self._evaluate_slide_transitions(content),
            "formatting_consistency": self._evaluate_formatting_consistency(content),
            "content_organization": self._evaluate_content_organization(content)
        })

        # Calculate overall score
        weights = {
            "relevance": 0.2,
            "slide_structure": 0.2,
            "visual_elements": 0.15,
            "slide_transitions": 0.1,
            "formatting_consistency": 0.15,
            "content_organization": 0.2
        }

        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)

        evaluation["score"] = weighted_sum / sum(weights.values())

        return evaluation

    def _evaluate_slide_structure(self, content: str) -> float:
        """Evaluate the structure of slides"""
        # Check for clear slide delineation
        slide_pattern = r"(Slide\s+\d+|#{1,3}\s+Slide\s+\d+)"
        slides = re.findall(slide_pattern, content, re.IGNORECASE)

        # Count unique slide references
        unique_slides = len(set(slides))

        if unique_slides == 0:
            return 0.0

        # Check for proper slide components (title, content)
        title_pattern = r"(Title:|#{1,3}.*)"
        titles = re.findall(title_pattern, content)

        # Check for content sections
        content_sections = re.split(slide_pattern, content)[1:]  # Skip the first empty section
        non_empty_sections = sum(1 for section in content_sections if len(section.strip()) > 50)

        # Scoring based on structure
        title_score = min(1.0, len(titles) / unique_slides)
        content_score = min(1.0, non_empty_sections / unique_slides)

        return (title_score * 0.4 + content_score * 0.6)

    def _evaluate_visual_elements(self, content: str) -> float:
        """Evaluate the quality and quantity of visual elements suggested"""
        # Check for image references
        image_pattern = r"(image|picture|photo|diagram|chart|graph|figure|visualization)"
        image_matches = re.findall(image_pattern, content, re.IGNORECASE)

        # Check for specific visual descriptions
        visual_desc_pattern = r"(showing|displaying|illustrating|depicting|representing)"
        visual_desc_matches = re.findall(visual_desc_pattern, content, re.IGNORECASE)

        # Calculate score based on visual element density
        content_length = len(content.split())
        if content_length == 0:
            return 0.0

        image_density = min(1.0, len(image_matches) / (content_length / 100))
        desc_quality = min(1.0, len(visual_desc_matches) / (len(image_matches) + 1))

        return (image_density * 0.7 + desc_quality * 0.3)

    def _evaluate_slide_transitions(self, content: str) -> float:
        """Evaluate logical transitions between slides"""
        # Look for transition words/phrases
        transition_patterns = [
            r"next",
            r"following",
            r"moving on",
            r"furthermore",
            r"additionally",
            r"in addition",
            r"consequently",
            r"as a result",
            r"therefore",
            r"thus",
            r"in conclusion"
        ]

        transition_count = 0
        for pattern in transition_patterns:
            matches = re.findall(r"\b" + pattern + r"\b", content, re.IGNORECASE)
            transition_count += len(matches)

        # Count slide breaks
        slide_breaks = len(re.findall(r"(Slide\s+\d+|#{1,3}\s+Slide)", content, re.IGNORECASE))

        if slide_breaks <= 1:
            return 0.5  # Not enough slides to evaluate transitions

        # Calculate ratio of transitions to slide breaks
        transition_ratio = min(1.0, transition_count / (slide_breaks - 1))

        return transition_ratio

    def _evaluate_formatting_consistency(self, content: str) -> float:
        """Evaluate consistency in formatting across slides"""
        # Check for consistent slide numbering
        slide_numbers = re.findall(r"Slide\s+(\d+)", content, re.IGNORECASE)

        if not slide_numbers:
            return 0.5  # Can't evaluate consistency without slide numbers

        # Check if numbers are sequential
        try:
            numbers = [int(num) for num in slide_numbers]
            expected_sequence = list(range(min(numbers), min(numbers) + len(numbers)))
            sequence_match = numbers == expected_sequence
        except ValueError:
            sequence_match = False

        # Check for consistent title formatting
        title_formats = re.findall(r"(#{1,3}\s+.*|Title:.*)", content, re.MULTILINE)
        title_pattern_counts = {}
        for title in title_formats:
            pattern = title[:3]  # Use first few chars as the pattern
            title_pattern_counts[pattern] = title_pattern_counts.get(pattern, 0) + 1

        dominant_pattern = max(title_pattern_counts.values()) if title_pattern_counts else 0
        total_titles = sum(title_pattern_counts.values())

        title_consistency = dominant_pattern / total_titles if total_titles > 0 else 0.5

        return (0.4 * int(sequence_match) + 0.6 * title_consistency)

    def _evaluate_content_organization(self, content: str) -> float:
        """Evaluate the logical organization and flow of content"""
        # Check for clear introduction
        has_intro = bool(re.search(r"\b(introduction|overview|agenda)\b", content[:500], re.IGNORECASE))

        # Check for clear conclusion
        has_conclusion = bool(re.search(r"\b(conclusion|summary|takeaway|key points)\b", content[-500:], re.IGNORECASE))

        # Check for section headers or topic transitions
        section_headers = re.findall(r"(#{1,3}\s+[^S].*)", content)  # Exclude slide headers

        # Check for bullet points indicating organized lists
        bullet_points = re.findall(r"(\*|\-|\d+\.)\s+", content)

        # Calculate score components
        structure_score = (0.25 * int(has_intro) +
                          0.25 * int(has_conclusion) +
                          0.25 * min(1.0, len(section_headers) / 5) +
                          0.25 * min(1.0, len(bullet_points) / 10))

        return structure_score

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for the PPT category"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}

        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}

        # Extract metrics from each result
        metrics = [
            "relevance",
            "slide_structure",
            "visual_elements",
            "slide_transitions",
            "formatting_consistency",
            "content_organization",
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