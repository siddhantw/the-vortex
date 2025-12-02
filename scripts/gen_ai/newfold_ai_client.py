"""
Newfold AI API Client
Replaces OpenAI API calls with Newfold AI endpoints using UJWT authentication
"""

import os
import json
import uuid
import requests
import logging
from typing import Dict, List, Optional, Any, Union
import jwt
import datetime

logger = logging.getLogger(__name__)

class NewfoldAIClient:
    """Client for Newfold AI API that mimics OpenAI interface"""

    # Newfold AI endpoints
    CHAT_ENDPOINT = "https://api-gw.uat.builderservices.io/ai-api/v1.0/chat"
    COMPLETIONS_ENDPOINT = "https://api-gw.uat.builderservices.io/ai-api/v1.0/completions"
    IMAGES_ENDPOINT = "https://api-gw.uat.builderservices.io/ai-api/v1.0/images/generations"

    # UJWT public key info for backend signing
    UJWT_PUBLIC_KEY = {
        "kty": "RSA",
        "e": "AQAB",
        "use": "sig",
        "kid": "jKNNvHvRVpML6_C6ntWLoHzl8a8XQQQHxuGNn0ATC_Y",
        "alg": "RS256",
        "n": "93_sgpl-5iCQRCckgt0Om2aLNaBZAJFQD6yvKmTC23Jc0KuOo9mFHGvFzxfb412zQkUXb_SQz8uqqNybghxVs66Va0MEno_4seFIn3HTLlU75iMFM11V_v3Dc8uP4XdNQXgQSts782KZ41iw0fmYm1YuwHr5iAjln9Hx_s-pK8nNXLg2K08J4cxl6jznuEMoxR2d-eM-QHSZrdX0lpKcHgA9ePzKoFKQ48Qh4q7FsEMIkjPUFZyRmBAHJfZ4k7Xb1ZMLLAT4mVt7pwl1Q_Ng-cudFCPdO-XzSy6rR9opuWNCS7KseZMc0N04ccXm5mRPInHKzX50bW70oM5nJwRlOQ"
    }

    def __init__(self, ujwt_token: Optional[str] = None, private_key: Optional[str] = None):
        """
        Initialize Newfold AI client

        Args:
            ujwt_token: Pre-generated UJWT token for authentication
            private_key: Private key for generating UJWT (if token not provided)
        """
        self.ujwt_token = ujwt_token or os.getenv("NEWFOLD_UJWT_TOKEN")
        self.private_key = private_key or os.getenv("NEWFOLD_PRIVATE_KEY")

        if not self.ujwt_token and not self.private_key:
            raise ValueError("Either UJWT token or private key must be provided")

    def _generate_ujwt(self) -> str:
        """Generate UJWT token using private key"""
        if not self.private_key:
            raise ValueError("Private key required to generate UJWT")

        # JWT payload
        payload = {
            "iss": "jarvis-test-automation",
            "aud": "api-gw.uat.builderservices.io",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            "iat": datetime.datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }

        # Generate JWT token
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        return token

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests"""
        token = self.ujwt_token or self._generate_ujwt()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Newfold AI API"""
        headers = self._get_auth_headers()

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Newfold AI API request failed: {e}")
            raise Exception(f"Newfold AI API error: {e}")

    def chat_completion_create(self, model: str, messages: List[Dict[str, str]],
                              temperature: float = 1.0, max_tokens: Optional[int] = None,
                              top_p: float = 1.0, n: int = 1, stop: Optional[List[str]] = None,
                              frequency_penalty: float = 0, presence_penalty: float = 0,
                              prompt_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create chat completion using Newfold AI API

        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            prompt_id: Prompt ID for Newfold AI
            metadata: Additional metadata

        Returns:
            Response in OpenAI-compatible format
        """
        # Map OpenAI model names to Newfold AI models
        model_mapping = {
            "gpt-3.5-turbo": "chat-ai-gpt-35-turbo",
            "gpt-4": "chat-ai-gpt-4",
            "gpt-4-turbo": "chat-ai-gpt-4-turbo"
        }

        newfold_model = model_mapping.get(model, "chat-ai-gpt-35-turbo")

        payload = {
            "PromptId": prompt_id or str(uuid.uuid4()),
            "MetaData": metadata or {"source": "jarvis-test-automation"},
            "OpenAiChatOptions": {
                "messages": messages,
                "Temperature": temperature,
                "TopP": top_p,
                "N": n,
                "Model": newfold_model
            }
        }

        if max_tokens:
            payload["OpenAiChatOptions"]["MaxTokens"] = max_tokens
        if stop:
            payload["OpenAiChatOptions"]["Stop"] = stop
        if frequency_penalty:
            payload["OpenAiChatOptions"]["FrequencyPenalty"] = frequency_penalty
        if presence_penalty:
            payload["OpenAiChatOptions"]["PresencePenalty"] = presence_penalty

        response = self._make_request(self.CHAT_ENDPOINT, payload)

        # Convert to OpenAI-compatible format
        return self._convert_to_openai_format(response)

    def completion_create(self, model: str, prompt: str, temperature: float = 1.0,
                         max_tokens: Optional[int] = None, top_p: float = 1.0,
                         n: int = 1, stop: Optional[List[str]] = None,
                         prompt_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create completion using Newfold AI API

        Args:
            model: Model name to use
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            prompt_id: Prompt ID for Newfold AI
            metadata: Additional metadata

        Returns:
            Response in OpenAI-compatible format
        """
        payload = {
            "PromptId": prompt_id or str(uuid.uuid4()),
            "MetaData": metadata or {"source": "jarvis-test-automation"},
            "OpenAiCompletionOptions": {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "model": model
            }
        }

        if max_tokens:
            payload["OpenAiCompletionOptions"]["max_tokens"] = max_tokens
        if stop:
            payload["OpenAiCompletionOptions"]["stop"] = stop

        response = self._make_request(self.COMPLETIONS_ENDPOINT, payload)
        return self._convert_to_openai_format(response)

    def image_generation_create(self, prompt: str, n: int = 1, size: str = "1024x1024",
                               prompt_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create image generation using Newfold AI API

        Args:
            prompt: Image generation prompt
            n: Number of images to generate
            size: Image size
            prompt_id: Prompt ID for Newfold AI
            metadata: Additional metadata

        Returns:
            Response in OpenAI-compatible format
        """
        payload = {
            "PromptId": prompt_id or str(uuid.uuid4()),
            "MetaData": metadata or {"source": "jarvis-test-automation"},
            "OpenAiImageOptions": {
                "prompt": prompt,
                "n": n,
                "size": size
            }
        }

        response = self._make_request(self.IMAGES_ENDPOINT, payload)
        return self._convert_to_openai_format(response)

    def _convert_to_openai_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Newfold AI response to OpenAI-compatible format"""
        # This conversion depends on the actual Newfold AI response format
        # For now, assuming it returns in a similar format to OpenAI
        # You may need to adjust this based on actual API responses

        if "choices" in response:
            return response

        # If the response format is different, convert it here
        # This is a placeholder implementation
        return {
            "choices": [
                {
                    "message": {
                        "content": response.get("content", response.get("text", "")),
                        "role": "assistant"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "created": response.get("created", int(datetime.datetime.utcnow().timestamp())),
            "id": response.get("id", str(uuid.uuid4())),
            "model": response.get("model", "newfold-ai"),
            "object": "chat.completion",
            "usage": response.get("usage", {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0
            })
        }


# Global client instance
_client = None

def get_client() -> NewfoldAIClient:
    """Get global Newfold AI client instance"""
    global _client
    if _client is None:
        _client = NewfoldAIClient()
    return _client

# OpenAI-compatible interface for easy replacement
class ChatCompletion:
    @staticmethod
    def create(**kwargs):
        return get_client().chat_completion_create(**kwargs)

class Completion:
    @staticmethod
    def create(**kwargs):
        return get_client().completion_create(**kwargs)

class Image:
    @staticmethod
    def create(**kwargs):
        return get_client().image_generation_create(**kwargs)

# Set global API key (for compatibility)
api_key = None

def set_api_key(key: str):
    """Set API key (for compatibility with OpenAI interface)"""
    global api_key
    api_key = key
    # Note: Newfold AI uses UJWT, so this is mainly for interface compatibility
