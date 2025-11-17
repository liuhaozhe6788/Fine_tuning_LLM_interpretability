"""
Configuration class for code generation using the teacher model (GPT-5).
"""
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class CodeGenerationConfig:
    """
    Configuration class for code generation parameters.
    
    This class centralizes all parameters used during code generation
    with the teacher model (GPT-5).
    """
    # Model configuration
    model_name: str = "gpt-5"
    
    # Generation parameters
    max_completion_tokens: int = 1000
    top_p: Optional[float] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Code extraction configuration
    code_start_marker: str = "###Python"
    code_end_marker: str = "###End Python"  
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_name": self.model_name,
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "code_start_marker": self.code_start_marker,
            "code_end_marker": self.code_end_marker
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "CodeGenerationConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            CodeGenerationConfig instance
        """
        filtered_dict = {k: v for k, v in config_dict.items() 
                        if v is not None}
        return cls(**filtered_dict)
    
    def get_openai_params(self) -> dict:
        """
        Get parameters formatted for OpenAI API call.
        
        Returns:
            Dictionary of parameters for OpenAI API
        """
        params = {
            "model": self.model_name,
            "max_completion_tokens": self.max_completion_tokens,
        }
        
        # Add optional parameters if they are set
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        
        return params


# Default configuration instance
default_config = CodeGenerationConfig()

