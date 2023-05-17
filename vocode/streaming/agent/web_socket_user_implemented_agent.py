import asyncio
import json
import logging
import websockets
from websockets.client import WebSocketClientProtocol

from typing import Awaitable, Callable, Optional, cast
from vocode.streaming.agent.base_agent import AgentMessageResponse, BaseAgent, StopAgentMessageResponse, TextAgentMessageResponse, TextAndStopAgentMessageResponse
from vocode.streaming.models.web_socket_agent import (
    AgentConfig,
    WebSocketAgentMessage,
    WebSocketAgentMessageEncoder,
    WebSocketAgentMessageType,
    WebSocketAgentTextMessage,
    WebSocketAgentTextStopMessage,
    WebSocketUserImplementedAgentConfig,
)

NUM_RESTARTS = 5


class WebSocketUserImplementedAgent(BaseAgent):

    def __init__(
        self,
        agent_config: WebSocketUserImplementedAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_config = agent_config
        self.on_response: Optional[Callable[[AgentMessageResponse], Awaitable]] = None
        self.logger = logger or logging.getLogger(__name__)

        self.has_ended = False
        self.agent_request_queue: asyncio.Queue = asyncio.Queue()

    def get_agent_config(self) -> WebSocketUserImplementedAgentConfig:
        return self.agent_config

    def set_on_response(self, on_response: Callable[[AgentMessageResponse], Awaitable]) -> None:
        self.on_response = on_response

    async def run(self) -> None:
        restarts = 0
        self.logger.info("Starting Socket Agent")
        while not self.has_ended and restarts < NUM_RESTARTS:
            await self._process()
            restarts += 1
            self.logger.debug(
                "Socket Agent connection died, restarting, num_restarts: %s", restarts
            )

    def send_transcript(
            self,
            transcription: str,
            is_interrupt: bool = False,
            conversation_id: str = ""
            ) -> None:
        self.logger.info("Sending transcript to Socket Agent: %s", transcription)
        message = WebSocketAgentTextMessage.from_text(text=transcription, conversation_id=conversation_id)
        self.agent_request_queue.put_nowait(message)

    def handle_incoming_socket_message(self, message: WebSocketAgentMessage) -> None:
        self.logger.info("Handling incoming message from Socket Agent: %s", message)

        agent_response: AgentMessageResponse
        if cast(WebSocketAgentTextMessage, message):
            agent_response = TextAgentMessageResponse(text=message.data.text)
        elif cast(WebSocketAgentTextStopMessage, message):
            agent_response = TextAndStopAgentMessageResponse(text=message.data.text)
            self.has_ended = True
        elif cast(WebSocketAgentTextStopMessage, message):
            agent_response = StopAgentMessageResponse()
            self.has_ended = True
        else:
            raise Exception("Unknown Socket message type")

        if self.on_response:
            self.on_response(agent_response)

    async def _process(self) -> None:
        extra_headers: dict[str, str] = {}
        socket_url = self.get_agent_config().respond.url
        self.logger.info("Connecting to web socket agent %s", socket_url)

        async with websockets.connect(
            socket_url,
            extra_headers=extra_headers
        ) as ws:
            async def sender(ws: WebSocketClientProtocol) -> None:  # sends audio to websocket
                while not self.has_ended:
                    self.logger.info("Waiting for data from agent request queue")
                    try:
                        if not self.agent_request_queue.empty():
                            self.logger.info("Request queue is not empty")
                        else:
                            self.logger.info("Request queue is empty")

                        agent_request = await asyncio.wait_for(self.agent_request_queue.get(), None)
                        agent_request_json = json.dumps(agent_request.to_json_dictionary())
                        self.logger.info(f"Sending data to web socket agent: {agent_request_json}")
                        if isinstance(agent_request, StopAgentMessageResponse):
                            self.has_ended = True
                        elif isinstance(agent_request, TextAndStopAgentMessageResponse):
                            # In practice, it doesn't make sense for the client to send a text and stop message to the agent service
                            self.has_ended = True

                    except asyncio.exceptions.TimeoutError:
                        break

                    await ws.send(agent_request_json)
                self.logger.debug("Terminating web socket agent sender")

            async def receiver(ws: WebSocketClientProtocol) -> None:
                while not self.has_ended:
                    try:
                        msg = await ws.recv()
                        self.logger.info(f"Received data from web socket agent: {msg}")
                    except Exception as e:
                        self.logger.debug(f"Got error \"{e}\" in Web Socket User Implemented Agent receiver")
                        break
                    data = json.loads(msg)

                    self.logger.info(f"Received json data from web socket agent: {data}")
                    message = WebSocketAgentMessage.parse_obj(data)
                    self.handle_incoming_socket_message(message)

                self.logger.debug("Terminating Web Socket User Implemented Agent receiver")

            await asyncio.gather(sender(ws), receiver(ws))

    def terminate(self):
        self.agent_request_queue.put_nowait(StopAgentMessageResponse())
