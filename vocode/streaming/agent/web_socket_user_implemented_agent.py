import asyncio
import json
import logging
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.utils.worker import InterruptibleEvent
import websockets
from websockets.client import WebSocketClientProtocol

from typing import Awaitable, Callable, Optional, cast
from vocode.streaming.agent.base_agent import (
    AgentResponse,
    AgentResponseMessage,
    BaseAsyncAgent,
    OneShotAgentResponse,
    StopAgentResponseMessage,
    TextAgentResponseMessage,
    TextAndStopAgentResponseMessage,
)
from vocode.streaming.models.websocket_agent import (
    WebSocketAgentMessage,
    WebSocketAgentStopMessage,
    WebSocketAgentTextMessage,
    WebSocketAgentTextStopMessage,
    WebSocketUserImplementedAgentConfig,
)

NUM_RESTARTS = 5


class WebSocketUserImplementedAgent(BaseAsyncAgent[WebSocketUserImplementedAgentConfig]):
    input_queue: asyncio.Queue[InterruptibleEvent[Transcription]]
    output_queue: asyncio.Queue[InterruptibleEvent[AgentResponse]]

    def __init__(
        self,
        agent_config: WebSocketUserImplementedAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_config = agent_config
        self.logger = logger or logging.getLogger(__name__)

        self.has_ended = False
        super().__init__(agent_config=agent_config, logger=logger)

    def get_agent_config(self) -> WebSocketUserImplementedAgentConfig:
        return self.agent_config

    async def _run_loop(self) -> None:
        restarts = 0
        self.logger.info("Starting Socket Agent")
        while not self.has_ended and restarts < NUM_RESTARTS:
            await self._process()
            restarts += 1
            self.logger.debug(
                "Socket Agent connection died, restarting, num_restarts: %s", restarts
            )

    def _handle_incoming_socket_message(self, message: WebSocketAgentMessage) -> None:
        self.logger.info("OUTPUT: Handling incoming message from Socket Agent: %s", message)

        agent_response_message: AgentResponseMessage

        if isinstance(message, WebSocketAgentTextMessage):
            agent_response_message = TextAgentResponseMessage(text=message.data.text)
        elif isinstance(message, WebSocketAgentTextStopMessage):
            agent_response_message = TextAndStopAgentResponseMessage(text=message.data.text)
            self.has_ended = True
        elif isinstance(message, WebSocketAgentStopMessage):
            agent_response_message = StopAgentResponseMessage()
            self.has_ended = True
        else:
            raise Exception("Unknown Socket message type")

        # Why is it necessary to upcast here?
        agent_response = cast(AgentResponse, OneShotAgentResponse(agent_response_message))

        # Change this to new agent response format
        interruptible_event = InterruptibleEvent(
            is_interruptible=self.get_agent_config().allow_agent_to_be_cut_off,
            payload=agent_response,
        )
        self.logger.info("Putting interruptible agent response event in output queue")
        self.output_queue.put_nowait(interruptible_event)

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
                        if not self.input_queue.empty():
                            self.logger.info("Request queue is not empty")
                        else:
                            self.logger.info("Request queue is empty")

                        transcription = await asyncio.wait_for(self.input_queue.get(), None)
                        self.logger.info("Transcript name: %s", transcription.payload.message)
                        agent_request = WebSocketAgentTextMessage.from_text(transcription.payload.message)
                        agent_request_json = json.dumps(agent_request.to_json_dictionary())
                        self.logger.info(f"Sending data to web socket agent: {agent_request_json}")
                        if isinstance(agent_request, (StopAgentResponseMessage, TextAndStopAgentResponseMessage)):
                            # In practice, it doesn't make sense for the client to send a text and stop message to the agent service
                            self.has_ended = True

                    except asyncio.exceptions.TimeoutError:
                        break

                    except Exception as e:
                        self.logger.error(f"WebSocket Agent Send Error: \"{e}\" in Web Socket User Implemented Agent sender")
                        break

                    await ws.send(agent_request_json)
                self.logger.debug("Terminating web socket agent sender")

            async def receiver(ws: WebSocketClientProtocol) -> None:
                while not self.has_ended:
                    try:
                        msg = await ws.recv()
                        self.logger.info("Received data from web socket agent")
                        data = json.loads(msg)
                        message = WebSocketAgentMessage.parse_obj(data)
                        self._handle_incoming_socket_message(message)

                    except websockets.exceptions.ConnectionClosed as e:
                        self.logger.error(f"WebSocket Agent Receive Error: Connection Closed - \"{e}\"")
                        break

                    except websockets.exceptions.ConnectionClosedOK as e:
                        self.logger.error(f"WebSocket Agent Receive Error: Connection Closed OK - \"{e}\"")
                        break

                    except websockets.exceptions.InvalidStatus as e:
                        self.logger.error(f"WebSocket Agent Receive Error: Invalid Status - \"{e}\"")

                    except Exception as e:
                        self.logger.error(f"WebSocket Agent Receive Error: \"{e}\"")
                        break

                self.logger.debug("Terminating Web Socket User Implemented Agent receiver")

            await asyncio.gather(sender(ws), receiver(ws))

    def terminate(self):
        self.output_queue.put_nowait(StopAgentResponseMessage())
        super().terminate()
