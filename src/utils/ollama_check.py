"""
Ollama service connectivity check
"""

import logging
import urllib.request
import urllib.error
from typing import Optional

from src.config.settings import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaHealthCheck:
    """Check Ollama service health"""
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize health check
        
        Args:
            config: Ollama configuration
        """
        self.config = config
        self.health_endpoint = f"{config.base_url}/api/tags"
    
    def check_connection(self) -> bool:
        """
        Check if Ollama service is accessible
        
        Returns:
            True if service is accessible, False otherwise
        """
        try:
            req = urllib.request.Request(self.health_endpoint)
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status == 200
                if status:
                    logger.info("Ollama service is accessible")
                return status
        except urllib.error.URLError as e:
            logger.warning(f"Cannot connect to Ollama service: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama: {e}")
            return False
    
    def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a specific model is available
        
        Args:
            model_name: Name of the model to check
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            req = urllib.request.Request(self.health_endpoint)
            with urllib.request.urlopen(req, timeout=5) as response:
                import json
                data = json.loads(response.read().decode())
                models = [model.get("name", "") for model in data.get("models", [])]
                available = model_name in models
                
                if available:
                    logger.info(f"Model {model_name} is available")
                else:
                    logger.warning(f"Model {model_name} is not available")
                
                return available
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
