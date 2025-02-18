from abc import ABC, abstractmethod
from anthropic import Anthropic
from openai import OpenAI
import os
import logging
import base64

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgiClient(ABC):
    @abstractmethod
    def get_response(self, model_input) -> str:
        pass

    @abstractmethod
    def get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        pass


class AnthropicClient(AgiClient):
    def __init__(self):
        logger.info("Creating Anthropic Client...")

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=key)

    def get_response(self, model_input, prompt_caching:bool = False) -> str:
        try:
            message = self.client.messages.create(
                model=model_input.get("model", "claude-3-5-sonnet-20241022"),
                max_tokens=model_input.get("max_tokens", 1000),
                temperature=model_input.get("temperature", 0),
                system=model_input.get("system"),
                messages=model_input.get("messages"),
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"} if prompt_caching else None
            )
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

    def get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        """Upload pdf to Claude.ai as part of prompt content"""

        if pdf_path:
            with open(pdf_path, "rb") as pdf_file:
                binary_data = pdf_file.read()
                base64_encoded_data = base64.standard_b64encode(binary_data)
                base64_pdf_string = base64_encoded_data.decode("utf-8")

            return [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_pdf_string,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        return prompt


class OpenAiClient(AgiClient):
    def __init__(self):
        logger.info("Creating OpenAI Client...")

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        OpenAI.api_key = key
        self.client = OpenAI()

    def get_response(self, model_input) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=model_input.get("model", "gpt-3.5-turbo"),
                max_tokens=model_input.get("max_tokens", 1000),
                temperature=model_input.get("temperature", 0),
                system=model_input.get("system"),
                messages=model_input.get("messages"),
            )
            response = (
                completion.choices[0].message
                if len(completion.choices) > 0
                else "I have no answer."
            )
            logger.debug(f"OpenAI response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {e}")
            return "Error getting response from OpenAI."

    # TODO: Implement PDF upload for OpenAI
    def get_pdf_prompt_content(self, prompt: str, pdf_path: str = None) -> str:
        """Upload pdf to OpenAI as part of prompt content"""
        return prompt


class AgiClientFactory:
    @staticmethod
    def create_client(client_type) -> AgiClient:
        if client_type == "anthropic":
            return AnthropicClient()
        elif client_type == "openai":
            return OpenAiClient()
        else:
            raise ValueError(f"Unknown client type: {client_type}")
