
# evaluators/formatting_evaluator.py
from typing import Dict, List, Any
import re
from evaluators.base_evaluator import BaseEvaluator

class FormattingEvaluator(BaseEvaluator):
    """Evaluator for text formatting capabilities"""
    
    def evaluate(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response for formatting metrics"""
        evaluation = super().evaluate(response, test_case)
        
        content = response.get("content", "")
        
        # Add formatting-specific metrics
        evaluation.update({
            "markdown_correctness": self._evaluate_markdown(content),
            "table_formatting": self._evaluate_tables(content),
            "list_formatting": self._evaluate_lists(content)
        })
        
        # Calculate overall score
        weights = {
            "relevance": 0.1,
            "markdown_correctness": 0.3,
            "table_formatting": 0.3,
            "list_formatting": 0.3
        }
        
        weighted_sum = sum(evaluation[metric] * weight
                          for metric, weight in weights.items()
                          if metric in evaluation)
                          
        evaluation["score"] = weighted_sum / sum(weights.values())
        
        return evaluation
    
    def _evaluate_markdown(self, content: str) -> float:
        """Evaluate correct markdown formatting"""
        # Look for markdown elements
        markdown_elements = {
            "headers": r"^#{1,6}\s+.+$",
            "bold": r"\*\*[^*]+\*\*",
            "italic": r"\*[^*]+\*|_[^_]+_",
            "code": r"`[^`]+`",
            "code_blocks": r"```[\s\S]*?```",
            "links": r"\[([^]]+)\]\(([^)]+)\)",
            "images": r"!\[([^]]*)\]\(([^)]+)\)",
            "blockquotes": r"^>\s+.+$",
            "horizontal_rule": r"^---$|^___$|^\*\*\*$"
        }
        
        # Count correct markdown elements
        markdown_count = 0
        valid_count = 0
        
        for element_type, pattern in markdown_elements.items():
            matches = re.findall(pattern, content, re.MULTILINE)
            count = len(matches)
            if count > 0:
                markdown_count += 1
                
                # Check validity of each match
                if element_type == "headers":
                    valid_headers = [m for m in matches if len(m.strip()) > 2]  # Header must have content
                    valid_count += len(valid_headers) / max(1, count)
                elif element_type == "links" or element_type == "images":
                    # For links/images, both text and URL should be present
                    if isinstance(matches[0], tuple) and all(len(m) == 2 for m in matches):
                        valid_count += 1
                    else:
                        valid_count += 0.5  # Partial validity
                else:
                    valid_count += 1  # Assume other elements are valid if present
        
        # No markdown used
        if markdown_count == 0:
            return 0.0
            
        # Calculate correctness score
        correctness_score = valid_count / markdown_count if markdown_count > 0 else 0
        
        # Bonus for diverse markdown usage
        diversity_ratio = markdown_count / len(markdown_elements)
        diversity_bonus = min(0.3, diversity_ratio * 0.3)
        
        return min(1.0, correctness_score + diversity_bonus)
    
    def _evaluate_tables(self, content: str) -> float:
        """Evaluate table formatting"""
        # Check for markdown tables
        table_pattern = r"\|(.+\|)+\s*\n\|([\s-]*\|)+"
        tables = re.findall(table_pattern, content)
        
        if not tables:
            # Check for ASCII art tables as fallback
            ascii_table_pattern = r"(\+[-+]+\+\s*\n\|[^|]+\|)"
            ascii_tables = re.findall(ascii_table_pattern, content)
            
            if not ascii_tables:
                return 0.0  # No tables found
                
            # Basic score for ASCII tables
            return 0.5
            
        # Evaluate table structure
        table_score = 0
        
        # Extract full tables with headers and rows
        full_tables = re.findall(r"\|(.+\|)+\s*\n\|([\s-]*\|)+\s*\n(\|.+\|[\s\S]*?(?=\n\n|\n$|$))", content)
        
        if full_tables:
            # Properly structured tables with headers and separators
            table_score = 1.0
        else:
            # Check for partial tables
            partial_tables = re.findall(r"\|(.+\|)+\s*\n(\|.+\|)", content)
            if partial_tables:
                # Tables without proper header separators
                table_score = 0.7
            else:
                # Very basic tables
                table_score = 0.4
                
        # Check for alignment indicators
        alignment_indicators = re.findall(r"\|[\s:-]*:[\s-]*\|", content)
        if alignment_indicators:
            table_score = min(1.0, table_score + 0.2)  # Bonus for alignment
            
        return table_score
    
    def _evaluate_lists(self, content: str) -> float:
        """Evaluate list formatting"""
        # Check for different list types
        list_patterns = {
            "unordered": r"^\s*[-*+]\s+.+$",
            "ordered": r"^\s*\d+\.\s+.+$",
            "nested_unordered": r"^\s+[-*+]\s+.+$",
            "nested_ordered": r"^\s+\d+\.\s+.+$",
            "task": r"^\s*[-*+]\s+\[[ x]\]\s+.+$"
        }
        
        # Count list items of each type
        list_counts = {}
        for list_type, pattern in list_patterns.items():
            matches = re.findall(pattern, content, re.MULTILINE)
            list_counts[list_type] = len(matches)
            
        # Calculate total list items
        total_items = sum(list_counts.values())
        if total_items == 0:
            return 0.0  # No lists found
            
        # Check list consistency
        list_sections = self._identify_list_sections(content)
        consistent_sections = 0
        
        for section in list_sections:
            lines = section.strip().split("\n")
            if len(lines) <= 1:
                continue
                
            # Check if all items use same marker type
            markers = [re.match(r"^\s*([-*+]|\d+\.)\s+", line) for line in lines]
            marker_types = [m.group(1)[0] if m else "" for m in markers]
            
            # Filter out empty markers
            marker_types = [m for m in marker_types if m]
            
            if marker_types and all(m == marker_types[0] for m in marker_types):
                consistent_sections += 1
                
        consistency_score = consistent_sections / len(list_sections) if list_sections else 0
        
        # Calculate diversity score (bonus for using different list types)
        used_types = sum(1 for count in list_counts.values() if count > 0)
        diversity_score = min(0.3, (used_types / len(list_patterns)) * 0.3)
        
        # Base score from list presence and count
        base_score = min(0.7, (total_items / 10) * 0.7)  # Assume 10 items is good
        
        return min(1.0, base_score + (consistency_score * 0.3) + diversity_score)
    
    def _identify_list_sections(self, content: str) -> List[str]:
        """Identify contiguous list sections in content"""
        lines = content.split("\n")
        list_sections = []
        current_section = []
        
        list_start_pattern = r"^\s*([-*+]|\d+\.)\s+"
        
        for line in lines:
            if re.match(list_start_pattern, line) and not current_section:
                # Start a new list section
                current_section = [line]
            elif re.match(list_start_pattern, line) or (current_section and line.strip().startswith(" ")):
                # Continue current list
                current_section.append(line)
            elif current_section:
                # End of current list
                list_sections.append("\n".join(current_section))
                current_section = []
                
        # Add last section if exists
        if current_section:
            list_sections.append("\n".join(current_section))
            
        return list_sections
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple test case results for formatting evaluation"""
        if not results:
            return {"score": 0.0, "error": "No results to aggregate"}
            
        # Filter out failed tests
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"score": 0.0, "error": "All tests failed"}
            
        # Extract metrics from each result
        metrics = [
            "relevance",
            "markdown_correctness",
            "table_formatting",
            "list_formatting",
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