from enum import Enum
import logging
import random
from typing import AsyncGenerator, Awaitable, Callable, Generator, Optional, Tuple
from vocode.streaming.models.agent import (
    AgentConfig,
    ChatGPTAgentConfig,
    LLMAgentConfig,
)
from vocode.streaming.models.model import TypedModel


class AgentMessageResponseType(str, Enum):
    BASE = "base_agent_message_response"
    TEXT = "text_agent_message_response"
    TEXT_AND_STOP = "text_and_stop_agent_message_response"
    STOP = "stop_agent_message_response"


class AgentMessageResponse(TypedModel, type=AgentMessageResponseType.BASE):
    conversation_id: Optional[str] = None


class TextAgentMessageResponse(TypedModel, type=AgentMessageResponseType.TEXT):
    text: str


class TextAndStopAgentMessageResponse(TypedModel, type=AgentMessageResponseType.TEXT_AND_STOP):
    text: str


class StopAgentMessageResponse(TypedModel, type=AgentMessageResponseType.STOP):
    pass


class BaseAgent:
    def __init__(
        self,
        agent_config: AgentConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.agent_config = agent_config
        self.on_response: Optional[Callable[[AgentMessageResponse], Awaitable]] = None
        self.logger = logger or logging.getLogger(__name__)

    def get_agent_config(self) -> AgentConfig:
        return self.agent_config

    # Validate it's safe to remove def start() in favor of run()
    async def run(self) -> None:
        pass

    def set_on_response(self, on_response: Callable[[AgentMessageResponse], Awaitable]):
        self.on_response = on_response

    def send_transcript(self, transcription: str, is_interrupt: bool = False, conversation_id: str = ""):
        pass

    def update_last_bot_message_on_cut_off(self, message: str):
        """Updates the last bot message in the conversation history when the human cuts off the bot's response."""
        pass

    def get_cut_off_response(self) -> Optional[str]:
        assert isinstance(self.agent_config, LLMAgentConfig) or isinstance(
            self.agent_config, ChatGPTAgentConfig
        )
        on_cut_off_messages = self.agent_config.cut_off_response.messages
        if on_cut_off_messages:
            return random.choice(on_cut_off_messages).text

    def terminate(self):
        pass
