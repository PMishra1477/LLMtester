# # clients/google_client.py
# import time
# from typing import Dict, Any, Tuple, Optional
# import google.generativeai as genai
# from clients.base_client import BaseClient
# from utils.logger import get_logger

# logger = get_logger(__name__)

# class GoogleClient(BaseClient):
#     """Client for Google models (Gemini)"""

#     def __init__(self, model: str, config: Dict[str, Any]):
#         super().__init__(model, config)
#         genai.configure(api_key=self.api_key)
#         self.model_obj = genai.GenerativeModel(model_name=self.model)

#     def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         """Execute a test with Google models"""
#         try:
#             prepared = self._prepare_prompt(test_case)
#             prompt = prepared["prompt"]
#             params = prepared["parameters"]

#             # Start timing
#             start_time = time.time()

#             # Configure generation parameters
#             generation_config = {
#                 "max_output_tokens": params.get("max_tokens", 1000),
#                 "temperature": params.get("temperature", 0.7),
#                 "top_p": params.get("top_p", 1.0),
#                 "top_k": params.get("top_k", 40)
#             }

#             # Make API call
#             response = self.model_obj.generate_content(
#                 prompt,
#                 generation_config=generation_config
#             )

#             # End timing
#             elapsed_time = time.time() - start_time

#             # Process response
#             content = response.text

#             # Estimate token usage (Google doesn't provide exact counts)
#             input_tokens = len(prompt) // 4  # Very rough estimate
#             output_tokens = len(content) // 4  # Very rough estimate

#             usage = {
#                 "input_tokens": input_tokens,
#                 "output_tokens": output_tokens,
#                 "total_tokens": input_tokens + output_tokens,
#                 "elapsed_time": elapsed_time
#             }

#             result = {
#                 "success": True,
#                 "content": content,
#                 "model": self.model,
#                 "provider": "google"
#             }

#             return result, usage

#         except Exception as e:
#             return self._handle_error(e, test_case)

#     def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
#         """Generate text from Google model - compatibility method for TestExecutor."""
#         try:
#             # Combine prompt and context if provided
#             full_prompt = prompt
#             if context:
#                 full_prompt = f"{context}\n\n{prompt}"
                
#             params = parameters or {}
            
#             # Configure generation parameters
#             generation_config = {
#                 "max_output_tokens": params.get("max_tokens", 1000),
#                 "temperature": params.get("temperature", 0.7),
#                 "top_p": params.get("top_p", 1.0),
#                 "top_k": params.get("top_k", 40)
#             }

#             response = self.model_obj.generate_content(
#                 full_prompt,
#                 generation_config=generation_config
#             )
#             return response.text
#         except Exception as e:
#             logger.error(f"Error generating text with {self.model}: {e}")
#             raise



# clients/google_client.py - REPLACE YOUR CURRENT FILE WITH THIS
import time
from typing import Dict, Any, Tuple, Optional
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from clients.base_client import BaseClient
from utils.logger import get_logger

logger = get_logger(__name__)

class GoogleClient(BaseClient):
    """Client for Google models (Gemini) with proper error handling"""

    def __init__(self, model: str, config: Dict[str, Any]):
        super().__init__(model, config)
        genai.configure(api_key=self.api_key)
        # Use version field if available, otherwise use model name
        self.model_id = config.get("version", model)
        self.model_obj = genai.GenerativeModel(model_name=self.model_id)
        
        logger.info(f"Initialized Google client for {model} (API model: {self.model_id})")

    def execute(self, test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a test with Google models"""
        try:
            prepared = self._prepare_prompt(test_case)
            prompt = prepared["prompt"]
            params = prepared["parameters"]

            # Start timing
            start_time = time.time()

            # Generate content safely
            content = self._generate_safely(prompt, params)

            # End timing
            elapsed_time = time.time() - start_time

            # Estimate token usage
            input_tokens = len(prompt) // 4
            output_tokens = len(content) // 4

            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "elapsed_time": elapsed_time
            }

            result = {
                "success": True,
                "content": content,
                "model": self.model,
                "api_model": self.model_id,
                "provider": "google"
            }

            return result, usage

        except Exception as e:
            return self._handle_error(e, test_case)

    def generate(self, prompt: str, context: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from Google model - compatibility method for TestExecutor."""
        try:
            # Combine prompt and context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
                
            params = parameters or {}
            
            return self._generate_safely(full_prompt, params)
            
        except Exception as e:
            logger.error(f"Error generating text with {self.model} (API: {self.model_id}): {e}")
            raise

    def _generate_safely(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate content with proper error handling for safety blocks"""
        try:
            # Configure generation parameters
            generation_config = {
                "max_output_tokens": params.get("max_tokens", 1000),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
                "top_k": params.get("top_k", 40)
            }

            # Configure safety settings to be less restrictive
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]

            # Make API call with retry for quota issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = 5 * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying {self.model_id} after {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)

                    response = self.model_obj.generate_content(
                        prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Extract content safely
                    return self._extract_content_safely(response)
                    
                except ResourceExhausted as e:
                    if "quota" in str(e).lower() and attempt < max_retries - 1:
                        logger.warning(f"Quota exceeded for {self.model_id}, retrying...")
                        continue
                    else:
                        logger.error(f"Quota exhausted for {self.model_id}")
                        return f"[Quota exceeded for {self.model_id} - please try again later]"
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error with {self.model_id}: {e}")
                        continue
                    else:
                        raise

        except Exception as e:
            logger.error(f"Failed to generate content with {self.model_id}: {e}")
            return f"[Error generating content with {self.model_id}: {str(e)}]"

    def _extract_content_safely(self, response) -> str:
        """Safely extract content from response, handling blocked content"""
        try:
            # First, check if response has candidates
            if not hasattr(response, 'candidates') or not response.candidates:
                logger.warning(f"No candidates in response for {self.model_id}")
                return "[No response generated - possibly blocked by safety filters]"

            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                
                if finish_reason == 2:  # SAFETY
                    logger.warning(f"Content blocked by safety filters for {self.model_id}")
                    return "[Content blocked by safety filters. This is a presentation generation request and should be safe. Please try rephrasing.]"
                elif finish_reason == 3:  # RECITATION
                    logger.warning(f"Content blocked due to recitation for {self.model_id}")
                    return "[Content blocked due to potential recitation. Please try a more specific or original request.]"
                elif finish_reason == 4:  # OTHER
                    logger.warning(f"Content blocked for other reasons for {self.model_id}")
                    return "[Content generation stopped. Please try modifying your request.]"
                elif finish_reason == 5:  # MAX_TOKENS
                    logger.info(f"Content truncated due to max tokens for {self.model_id}")
                    # Continue to extract partial content
            
            # Try to extract content from parts
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    if hasattr(candidate.content.parts[0], 'text'):
                        content = candidate.content.parts[0].text
                        if content and content.strip():
                            return content
            
            # Fallback: try the quick accessor (this is what was failing before)
            try:
                return response.text
            except ValueError as ve:
                if "finish_reason" in str(ve):
                    logger.warning(f"Response blocked for {self.model_id}: {ve}")
                    return "[Response blocked by content filters. Please try rephrasing your presentation request.]"
                else:
                    raise ve
            
        except Exception as e:
            logger.warning(f"Could not extract content from response for {self.model_id}: {e}")
            return f"[Error extracting content from {self.model_id}: {str(e)}]"

        # Final fallback
        return "[No content could be extracted from the response]"