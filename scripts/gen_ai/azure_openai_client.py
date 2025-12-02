"""
Azure OpenAI Client
Replaces Newfold AI with Azure OpenAI using the provided configuration
"""

import os
import logging
from typing import Dict, List, Optional, Any
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """Client for Azure OpenAI that provides a unified interface"""

    def __init__(self,
                 azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_version: str = "2024-10-21",
                 deployment_name: Optional[str] = None):
        """
        Initialize Azure OpenAI client

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
            deployment_name: Default deployment name
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        # Do NOT use any hardcoded default API key; require env var or explicit arg
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not self.azure_endpoint or not self.api_key:
            # Defer raising here to allow callers to check is_configured()
            logger.warning("Azure OpenAI not fully configured: missing endpoint or API key")
            self.client = None
        else:
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        
        # Add chat attribute for compatibility with standard OpenAI client interface
        self.chat = ChatCompletionsWrapper(self)

    def is_configured(self) -> bool:
        """Return True if the client has all required configuration and is usable."""
        return self.client is not None and bool(self.deployment_name)

    def chat_completion_create(self,
                              model: str = None,
                              messages: List[Dict[str, str]] = None,
                              temperature: float = 1.0,
                              max_tokens: Optional[int] = None,
                              top_p: float = 1.0,
                              n: int = 1,
                              stop: Optional[List[str]] = None,
                              frequency_penalty: float = 0,
                              presence_penalty: float = 0,
                              **kwargs) -> Dict[str, Any]:
        """
        Create chat completion using Azure OpenAI

        Args:
            model: Model/deployment name (uses default if not provided)
            messages: List of messages in chat format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty

        Returns:
            Response in OpenAI format
        """
        if not messages:
            raise ValueError("Messages are required for chat completion")

        if not self.is_configured():
            raise RuntimeError("Azure OpenAI client is not configured. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT.")

        # Use provided model or default deployment
        deployment = model or self.deployment_name
        if not deployment:
            raise ValueError("Model/deployment name must be provided")

        try:
            response = self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )

            # Convert to dict format for compatibility
            return {
                "choices": [
                    {
                        "message": {
                            "content": choice.message.content,
                            "role": choice.message.role
                        },
                        "index": choice.index,
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                } if response.usage else None,
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model
            }
        except Exception as e:
            logger.error(f"Azure OpenAI API request failed: {e}")
            raise Exception(f"Azure OpenAI API error: {e}")

    def completion_create(self,
                         model: str = None,
                         prompt: str = None,
                         temperature: float = 1.0,
                         max_tokens: Optional[int] = None,
                         top_p: float = 1.0,
                         n: int = 1,
                         stop: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Create completion using Azure OpenAI (converted to chat format)

        Args:
            model: Model/deployment name
            prompt: Text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            n: Number of completions to generate
            stop: Stop sequences

        Returns:
            Response in OpenAI format
        """
        if not prompt:
            raise ValueError("Prompt is required for completion")

        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]

        return self.chat_completion_create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop
        )

    def generate_response(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Simple helper method to generate a text response

        Args:
            prompt: Input prompt
            model: Model/deployment name
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        response = self.completion_create(model=model, prompt=prompt, **kwargs)
        return response["choices"][0]["message"]["content"]

    def generate_chat_response(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> str:
        """
        Simple helper method to generate a chat response

        Args:
            messages: List of chat messages
            model: Model/deployment name
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        response = self.chat_completion_create(model=model, messages=messages, **kwargs)
        return response["choices"][0]["message"]["content"]

class ChatCompletionsWrapper:
    """Wrapper to provide chat.completions interface for compatibility"""
    
    def __init__(self, azure_client):
        self.azure_client = azure_client
        self.completions = CompletionsWrapper(azure_client)

class CompletionsWrapper:
    """Wrapper to provide completions.create interface for compatibility"""
    
    def __init__(self, azure_client):
        self.azure_client = azure_client
    
    def create(self, model=None, messages=None, **kwargs):
        """Create method that matches OpenAI client interface"""
        return self.azure_client.chat_completion_create(
            model=model,
            messages=messages,
            **kwargs
        )