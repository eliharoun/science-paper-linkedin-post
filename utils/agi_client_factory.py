from abc import ABC, abstractmethod
from anthropic import Anthropic
from openai import OpenAI
import os
import logging
import base64
from ollama import GenerateResponse
from ollama import Client as OllamaClient
from .models.claude_models import model_names as claude_models
from .models.openai_models import model_names as openai_models
from .models.ollama_models import model_names as ollama_models
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgiClient(ABC):
    @abstractmethod
    def get_response(self, model_input) -> str:
        pass

    @abstractmethod
    def _get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        pass

    @abstractmethod
    def _get_prompt_content(self, prompt: str) -> str:
        pass


class AnthropicClient(AgiClient):
    """Client for interacting with the Anthropic API"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        logger.info("Creating Anthropic Client...")

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=key)
        self.model_name = model_name

    def get_response(self, model_input: dict, prompt_caching: bool = True) -> str:
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=10, max=60),
            reraise=True,
        )
        def _make_api_call(self, model_input, prompt_caching):
            return self.client.messages.create(
                model=self.model_name,
                max_tokens=model_input.get("max_tokens", 1000),
                temperature=model_input.get("temperature", 0),
                system=model_input.get("system"),
                messages=self._get_pdf_prompt_content(
                    prompt=model_input.get("prompt"),
                    pdf_path=model_input.get("pdf_path"),
                ),
                extra_headers=(
                    {"anthropic-beta": "prompt-caching-2024-07-31"}
                    if prompt_caching
                    else None
                ),
            )

        try:
            message = _make_api_call(self, model_input, prompt_caching)
            response = (
                message.content[0].text
                if len(message.content) > 0
                else "I have no answer."
            )
            logger.debug(f"Anthropic response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from Anthropic: {e}")
            return "Error getting response from Anthropic."

    def _get_prompt_content(self, prompt: str) -> str:
        return [{"role": "user", "content": prompt}]

    def _get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        """Upload pdf to Claude.ai as part of prompt content"""

        if pdf_path:
            with open(pdf_path, "rb") as pdf_file:
                binary_data = pdf_file.read()
                base64_encoded_data = base64.standard_b64encode(binary_data)
                base64_pdf_string = base64_encoded_data.decode("utf-8")

            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64_pdf_string,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return prompt


class OpenAiClient(AgiClient):
    """Client for interacting with the OpenAI API"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        logger.info("Creating OpenAI Client...")

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        OpenAI.api_key = key
        self.client = OpenAI()
        self.model_name = model_name

    def get_response(self, model_input: dict) -> str:
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=10, max=60),
            reraise=True,
        )
        def _make_api_call(self, model_input):
            return self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=model_input.get("max_tokens", 1000),
                temperature=model_input.get("temperature", 0),
                system=model_input.get("system"),
                messages=self._get_prompt_content(model_input.get("prompt")),
            )

        try:
            completion = _make_api_call(self, model_input)
            response = (
                completion.choices[0].message.content
                if len(completion.choices) > 0
                else "I have no answer."
            )
            logger.debug(f"OpenAI response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {e}")
            return "Error getting response from OpenAI."

    def _get_prompt_content(self, prompt: str) -> str:
        return [{"role": "user", "content": prompt}]

    # TODO: Implement PDF upload for OpenAI
    def _get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        """Upload pdf to OpenAI as part of prompt content"""
        return [{"role": "user", "content": prompt}]


class LocalModelClient(AgiClient):
    """Client for interacting with a local model using the Ollama library https://ollama.com/search"""

    def __init__(self, model_name: str = "llama3.2"):
        logger.info("Creating Local Model Client...")
        self.client = OllamaClient()
        self.model_name = model_name

    def get_response(self, model_input: dict) -> str:
        try:
            chat_response: GenerateResponse = self.client.generate(
                model=self.model_name,
                system=model_input.get("system"),
                prompt=model_input.get("prompt"),
                options={"temperature": model_input.get("temperature", 0)},
                stream=False,
            )
            response = (
                chat_response.response
                if chat_response.response
                else "I have no answer."
            )
            logger.debug(f"{model_input.get('model')} response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from local model: {e}")
            return "Error getting response from local model."

    def _get_prompt_content(self, prompt: str) -> str:
        return prompt

    # TODO: Implement PDF upload for local model
    def _get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        """Upload pdf to local model as part of prompt content"""
        return prompt


class AgiClientFactory:
    @staticmethod
    def create_client(client_type: str, model_name: str) -> AgiClient:
        if client_type == "anthropic" and model_name in claude_models:
            return AnthropicClient(model_name)
        elif client_type == "openai" and model_name in openai_models:
            return OpenAiClient(model_name)
        elif client_type == "local" and model_name in ollama_models:
            return LocalModelClient(model_name)
        else:
            raise ValueError(
                f"Unknown client type: {client_type} or model name: {model_name}. Or model not supported by the client."
            )
