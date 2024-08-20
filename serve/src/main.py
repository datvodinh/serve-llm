from typing import Dict, Any
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        pass

    def reconfigure(self, config: Dict[str, Any]):
        """
        Reconfigure the deployment with new settings.

        This method updates the engine arguments and associated configurations
        for the deployment. It initializes a new instance of AsyncLLMEngine
        based on the provided engine arguments and resets the chat serving state.

        Args:
            config : Dict[str, Any]
                A dictionary containing the configuration settings which include
                engine arguments under the "engine_args" key and optionally the
                response role under the "response_role" key.

        Returns:
            None

        Logs:
            Logs the engine arguments with which the deployment is being started.

        Example:
            ```python
            config = {
                "engine_args": {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "max_model_len": 8192,
                    ...
                },
                "response_role": "assistant"
            }
            deployment_instance.reconfigure(config)
            ```
        """
        engine_args = AsyncEngineArgs(**config["engine_args"])
        logger.info(f"Starting with engine args: {engine_args}")

        self.openai_serving_chat = None
        self.response_role = config.get("response_role", "assistant")
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """
        Handle the creation of chat completions.

        This endpoint processes incoming chat requests and generates responses
        using the configured language model. If the OpenAIServingChat instance 
        is not initialized, it sets it up with the appropriate configuration 
        from the engine.

        Args:
            request : ChatCompletionRequest
                The incoming chat completion request.
            raw_request : Request
                The raw HTTP request object.

        Returns:
            JSONResponse or StreamingResponse:
                A JSON response containing the chat completion if request.stream is False,
                otherwise a streaming response containing the chat event stream.
        
        Raises:
            Exception
                If any errors occur during the generation of the chat completion, 
                they will be logged and appropriate error responses will be returned.
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                async_engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                response_role=self.response_role,
                lora_modules=None,
                chat_template=None,
                prompt_adapters=None,
                request_logger=None,
            )

        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request=request,
            raw_request=raw_request,
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code,
            )
        if request.stream:
            return StreamingResponse(
                content=generator,
                media_type="text/event-stream",
            )
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


deployment = VLLMDeployment.bind()
