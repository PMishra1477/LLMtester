
import os
import json
import yaml
from typing import Dict, Any, Optional, List
from utils.logger import get_logger

logger = get_logger(__name__)

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Created directory: {directory}")

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file into a dictionary.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary with YAML contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"YAML file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise

def save_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        file_path: Path to save YAML file
        data: Dictionary to save
        
    Raises:
        yaml.YAMLError: If YAML serialization fails
    """
    ensure_directory_exists(os.path.dirname(file_path))
    
    try:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.debug(f"Saved YAML file: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file into a dictionary.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with JSON contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        raise

def save_json(file_path: str, data: Dict[str, Any]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        file_path: Path to save JSON file
        data: Dictionary to save
        
    Raises:
        TypeError: If data cannot be serialized to JSON
    """
    ensure_directory_exists(os.path.dirname(file_path))
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved JSON file: {file_path}")
    except TypeError as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise

def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """
    List files in a directory, optionally filtering by extension.
    
    Args:
        directory: Directory to list files from
        extension: Optional file extension to filter by (e.g., '.yaml')
        
    Returns:
        List of file paths
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if extension is None or file.endswith(extension):
                files.append(file_path)
    
    return files

def read_text_file(file_path: str) -> str:
    """
    Read text file contents.
    
    Args:
        file_path: Path to text file
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Text file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        raise

def write_text_file(file_path: str, content: str) -> None:
    """
    Write content to text file.
    
    Args:
        file_path: Path to text file
        content: Content to write
    """
    ensure_directory_exists(os.path.dirname(file_path))
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug(f"Saved text file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing text file {file_path}: {e}")
        raise