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
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
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
