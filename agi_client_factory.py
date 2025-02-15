from abc import ABC, abstractmethod
from anthropic import Anthropic
from openai import OpenAI
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Client(ABC):
    @abstractmethod
    def getResponse(self, model_input) -> str:
        pass


class AnthropicClient(Client):
    def __init__(self):
        logger.info("Creating Anthropic Client...")

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=key)

    def getResponse(self, model_input) -> str:
        try:
            message = self.client.messages.create(model_input)
            response = (
                message.content[0].text
                if len(message.content) > 0
                else "I have no answer."
            )
            logger.info(f"Anthropic response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from Anthropic: {e}")
            return "Error getting response from Anthropic."


class OpenAiClient(Client):
    def __init__(self):
        logger.info("Creating OpenAI Client...")

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        OpenAI.api_key = key
        self.client = OpenAI()

    def getResponse(self, model_input) -> str:
        try:
            completion = self.client.chat.completions.create(model_input)
            response = (
                completion.choices[0].message
                if len(completion.choices) > 0
                else "I have no answer."
            )
            logger.info(f"OpenAI response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {e}")
            return "Error getting response from OpenAI."


class AgiClientFactory:
    @staticmethod
    def create_client(client_type) -> Client:
        if client_type == "anthropic":
            return AnthropicClient()
        elif client_type == "openai":
            return OpenAiClient()
        else:
            raise ValueError(f"Unknown client type: {client_type}")
